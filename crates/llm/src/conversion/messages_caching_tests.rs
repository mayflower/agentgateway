use serde_json::{Value, json};

use super::*;

/// All boundaries on, no threshold or offset.
fn all() -> PromptCachingConfig {
	PromptCachingConfig {
		cache_system: true,
		cache_messages: true,
		cache_tools: true,
		min_tokens: None,
		cache_message_offset: 0,
	}
}

fn marker() -> Value {
	json!({"type": "ephemeral"})
}

fn two_messages() -> Value {
	json!([
		{"role": "user", "content": [{"type": "text", "text": "one"}]},
		{"role": "user", "content": [{"type": "text", "text": "two"}]},
	])
}

// --- native Messages shape -------------------------------------------------

#[test]
fn marks_system_tools_and_messages() {
	let mut req = json!({
		"system": [{"type": "text", "text": "sys"}],
		"messages": two_messages(),
		"tools": [{"name": "a"}, {"name": "b"}],
	});
	apply(&mut req, &all());

	assert_eq!(req["system"][0]["cache_control"], marker());
	// Default boundary is the second-to-last message.
	assert_eq!(req["messages"][0]["content"][0]["cache_control"], marker());
	assert!(
		req["messages"][1]["content"][0]
			.get("cache_control")
			.is_none()
	);
	// Tools are marked on the last tool only.
	assert!(req["tools"][0].get("cache_control").is_none());
	assert_eq!(req["tools"][1]["cache_control"], marker());
}

#[test]
fn converts_string_shorthand_to_blocks_only_when_marking() {
	let mut req = json!({
		"system": "you are helpful",
		"messages": [
			{"role": "user", "content": "one"},
			{"role": "user", "content": "two"},
		],
	});
	apply(&mut req, &all());

	// The marked values become single-text-block arrays preserving their text.
	assert_eq!(
		req["system"],
		json!([{"type": "text", "text": "you are helpful", "cache_control": {"type": "ephemeral"}}])
	);
	assert_eq!(
		req["messages"][0]["content"],
		json!([{"type": "text", "text": "one", "cache_control": {"type": "ephemeral"}}])
	);
	// The unmarked message keeps its original string form.
	assert_eq!(req["messages"][1]["content"], json!("two"));
}

#[test]
fn preserves_unknown_fields() {
	let mut req = json!({
		"system": [{"type": "text", "text": "sys", "unknown": 1}],
		"messages": two_messages(),
		"metadata": {"user_id": "u"},
	});
	apply(&mut req, &all());

	assert_eq!(req["system"][0]["unknown"], json!(1));
	assert_eq!(req["metadata"]["user_id"], json!("u"));
}

// --- explicit client markers ----------------------------------------------

#[test]
fn explicit_marker_is_never_overwritten_and_keeps_its_properties() {
	let mut req = json!({
		"system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
		"messages": two_messages(),
	});
	apply(&mut req, &all());

	// TTL and any other explicit properties survive untouched.
	assert_eq!(
		req["system"][0]["cache_control"],
		json!({"type": "ephemeral", "ttl": "1h"})
	);
}

#[test]
fn explicit_and_generated_markers_coexist() {
	let mut req = json!({
		"system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
		"messages": two_messages(),
	});
	apply(&mut req, &all());

	assert_eq!(req["system"][0]["cache_control"], marker());
	// The message boundary still gets its generated marker.
	assert_eq!(req["messages"][0]["content"][0]["cache_control"], marker());
}

#[test]
fn no_duplicate_marker_at_an_already_marked_boundary() {
	let mut req = json!({
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "one", "cache_control": {"type": "ephemeral"}}]},
			{"role": "user", "content": [{"type": "text", "text": "two"}]},
		],
	});
	let before = req.clone();
	apply(&mut req, &all());
	assert_eq!(req, before, "already-marked boundary must be left alone");
}

#[test]
fn four_explicit_markers_exhaust_the_budget() {
	let marked = |t: &str| json!({"type": "text", "text": t, "cache_control": {"type": "ephemeral"}});
	let mut req = json!({
		"system": [marked("s1"), marked("s2")],
		"messages": [
			{"role": "user", "content": [marked("m1")]},
			{"role": "user", "content": [marked("m2")]},
			{"role": "user", "content": [{"type": "text", "text": "current"}]},
		],
		"tools": [{"name": "a"}],
	});
	let before = req.clone();
	apply(&mut req, &all());

	assert_eq!(req, before, "no capacity left, so nothing may be added");
	assert!(req["tools"][0].get("cache_control").is_none());
}

#[test]
fn generated_markers_use_only_remaining_capacity() {
	let marked = |t: &str| json!({"type": "text", "text": t, "cache_control": {"type": "ephemeral"}});
	// Three explicit markers leave room for exactly one generated marker, which the
	// system boundary claims first.
	let mut req = json!({
		"system": [{"type": "text", "text": "sys"}],
		"messages": [
			{"role": "user", "content": [marked("m1")]},
			{"role": "user", "content": [marked("m2")]},
			{"role": "user", "content": [marked("m3")]},
			{"role": "user", "content": [{"type": "text", "text": "current"}]},
		],
		"tools": [{"name": "a"}],
	});
	apply(&mut req, &all());

	assert_eq!(req["system"][0]["cache_control"], marker());
	assert!(req["tools"][0].get("cache_control").is_none());
	assert!(
		req["messages"][2]["content"][0]["cache_control"] == marker(),
		"pre-existing marker untouched"
	);
}

// --- configuration semantics ----------------------------------------------

#[test]
fn disabled_boundaries_generate_nothing() {
	let cfg = PromptCachingConfig {
		cache_system: false,
		cache_messages: false,
		cache_tools: false,
		..all()
	};
	let mut req = json!({
		"system": [{"type": "text", "text": "sys"}],
		"messages": two_messages(),
		"tools": [{"name": "a"}],
	});
	let before = req.clone();
	apply(&mut req, &cfg);
	assert_eq!(req, before);
}

#[test]
fn min_tokens_gates_the_system_boundary_only() {
	let cfg = PromptCachingConfig {
		min_tokens: Some(100),
		..all()
	};
	let mut req = json!({
		"system": [{"type": "text", "text": "short"}],
		"messages": two_messages(),
	});
	apply(&mut req, &cfg);

	assert!(
		req["system"][0].get("cache_control").is_none(),
		"system below threshold must not be marked"
	);
	assert_eq!(
		req["messages"][0]["content"][0]["cache_control"],
		marker(),
		"minTokens must not gate the message boundary"
	);
}

#[test]
fn min_tokens_met_marks_system() {
	let cfg = PromptCachingConfig {
		min_tokens: Some(10),
		..all()
	};
	// 10 words * 1.3 = 13 estimated tokens, matching the Bedrock heuristic.
	let long = "a b c d e f g h i j";
	let mut req = json!({ "system": [{"type": "text", "text": long}], "messages": two_messages() });
	apply(&mut req, &cfg);
	assert_eq!(req["system"][0]["cache_control"], marker());
}

#[test]
fn cache_message_offset_walks_the_boundary_back_and_clamps() {
	let messages = || {
		json!([
			{"role": "user", "content": [{"type": "text", "text": "0"}]},
			{"role": "user", "content": [{"type": "text", "text": "1"}]},
			{"role": "user", "content": [{"type": "text", "text": "2"}]},
			{"role": "user", "content": [{"type": "text", "text": "3"}]},
		])
	};
	// offset 0 -> second-to-last (index 2); offset 1 -> index 1.
	for (offset, expected) in [(0, 2), (1, 1), (2, 0)] {
		let cfg = PromptCachingConfig {
			cache_system: false,
			cache_tools: false,
			cache_message_offset: offset,
			..all()
		};
		let mut req = json!({ "messages": messages() });
		apply(&mut req, &cfg);
		assert_eq!(
			req["messages"][expected]["content"][0]["cache_control"],
			marker(),
			"offset {offset} should mark index {expected}"
		);
	}
	// A larger offset clamps to the first message rather than underflowing.
	let cfg = PromptCachingConfig {
		cache_system: false,
		cache_tools: false,
		cache_message_offset: 99,
		..all()
	};
	let mut req = json!({ "messages": messages() });
	apply(&mut req, &cfg);
	assert_eq!(req["messages"][0]["content"][0]["cache_control"], marker());
}

#[test]
fn single_message_has_nothing_to_reuse() {
	let cfg = PromptCachingConfig {
		cache_system: false,
		cache_tools: false,
		..all()
	};
	let mut req =
		json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]});
	let before = req.clone();
	apply(&mut req, &cfg);
	assert_eq!(req, before);
}

// --- best-effort behaviour -------------------------------------------------

#[test]
fn missing_or_empty_boundaries_are_a_no_op_not_an_error() {
	for mut req in [
		json!({}),
		json!({"messages": []}),
		json!({"system": [], "messages": [], "tools": []}),
		json!({"messages": two_messages(), "tools": []}),
	] {
		apply(&mut req, &all());
	}
}

#[test]
fn unmarkable_content_is_skipped_rather_than_corrupted() {
	// A trailing non-object block cannot carry a marker; the last object block takes it.
	let mut req = json!({
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "one"}, "raw"]},
			{"role": "user", "content": [{"type": "text", "text": "two"}]},
		],
	});
	apply(&mut req, &all());
	assert_eq!(req["messages"][0]["content"][0]["cache_control"], marker());
	assert_eq!(req["messages"][0]["content"][1], json!("raw"));
}

#[test]
fn non_object_request_is_ignored() {
	let mut req = json!("not a request");
	apply(&mut req, &all());
	assert_eq!(req, json!("not a request"));
}

// --- block types that cannot carry a marker --------------------------------

#[test]
fn thinking_blocks_never_receive_a_marker() {
	// An assistant turn ending in a thinking block is normal with extended thinking, and
	// `thinking` has no cache_control field upstream. Walk back to the last markable block.
	let mut req = json!({
		"messages": [
			{"role": "assistant", "content": [
				{"type": "text", "text": "reasoned answer"},
				{"type": "thinking", "thinking": "...", "signature": "sig"},
			]},
			{"role": "user", "content": [{"type": "text", "text": "next"}]},
		],
	});
	apply(&mut req, &all());

	assert_eq!(req["messages"][0]["content"][0]["cache_control"], marker());
	assert!(
		req["messages"][0]["content"][1]
			.get("cache_control")
			.is_none(),
		"thinking block must never be marked"
	);
}

#[test]
fn redacted_thinking_only_message_is_skipped() {
	let mut req = json!({
		"messages": [
			{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]},
			{"role": "user", "content": [{"type": "text", "text": "next"}]},
		],
	});
	let before = req.clone();
	apply(&mut req, &all());
	assert_eq!(req, before, "no markable block, so the boundary is skipped");
}

#[test]
fn unknown_block_types_are_skipped_not_marked() {
	// Losing the optimization is harmless; emitting a marker Anthropic rejects is not.
	let mut req = json!({
		"messages": [
			{"role": "user", "content": [{"type": "some_future_block", "x": 1}]},
			{"role": "user", "content": [{"type": "text", "text": "next"}]},
		],
	});
	let before = req.clone();
	apply(&mut req, &all());
	assert_eq!(req, before);
}

#[test]
fn every_cacheable_block_type_can_be_marked() {
	for ty in CACHEABLE_BLOCK_TYPES {
		let mut req = json!({
			"messages": [
				{"role": "user", "content": [{"type": ty}]},
				{"role": "user", "content": [{"type": "text", "text": "next"}]},
			],
		});
		apply(&mut req, &all());
		assert_eq!(
			req["messages"][0]["content"][0]["cache_control"],
			marker(),
			"{ty} should accept a marker"
		);
	}
}

#[test]
fn tools_without_a_type_discriminator_are_still_marked() {
	// Custom Anthropic tools are {name, description, input_schema} with no `type`.
	let mut req = json!({
		"messages": two_messages(),
		"tools": [{"name": "a", "input_schema": {}}],
	});
	apply(&mut req, &all());
	assert_eq!(req["tools"][0]["cache_control"], marker());
}
