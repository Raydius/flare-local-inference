/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg(feature = "server")]
//! A request that spells no control tokenizes the same without the added vocabulary.
//!
//! The literal-token path encodes request text that spells a control with the tokenizer's added
//! vocabulary removed. Each checkpoint tokenizer must encode ordinary text identically both ways.
//! One test per checkpoint runs every check and fails when the variable naming the checkpoint is
//! unset. The controls are the special added tokens, the ones the chat template names, and the
//! markers that the checkpoint's own frontend contract names, as the server loads them; the other
//! added tokens (`/nothink`, fill-in-the-middle and file markers) are lexical, and request text
//! that spells them tokenizes to that token. Every declared marker of the contract (reasoning
//! delimiters and terminal markers, tool-call delimiters and argument tags, image tokens) must be
//! special or named by the template: a declared marker that is lexical fails the test. DeepSeek
//! V4.1 runs through its native encoder with no template, which tokenizes all request text as
//! ordinary text, so no added token is lexical there. Each checkpoint lists the request slots its
//! template rejects and the (slot, spelling) pairs it renders differently around the spelled
//! word; a slot or pair outside those lists that is skipped fails the test. A tokenizer whose
//! pre-tokenizer is Metaspace (it prepends a space) would tokenize a region re-encoded without the
//! added vocabulary differently from the first pass; none of the seven tokenizers tested here uses
//! one, and the tokenizers listed in docs/TODO.md are unread.

#[path = "support/templates.rs"]
mod template_fixtures;

use flare_helios::frontend::{Asset, Frontend, FrontendAssets};
use flare_helios::{frontend, model, scheduler};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
#[path = "support/frontend.rs"]
mod support;

const TEXT: &[&str] = &[
    "Plain sentence, numbers 3.14159 and 1,000,000. Ends here.",
    "line one\n\nline two\n\n\n\nline three\n",
    "a    b        c\t\td          e\r\n\r\nf",
    "<b>bold</b> <div class=\"x\">y</div> <br/> <p>para</p> <table><tr><td>1</td></tr></table>",
    "fn main() {\n    let x = vec![1, 2, 3];\n    println!(\"{x:?}\");\n}\n\tif (a < b && c > d) { return; }\n",
    "def f(x):\n\treturn {'k': [x, None]}  # comment\n{{ jinja }} {% raw %} [1] <1> </ <| |> <think <|im_ ",
    "na\u{ef}ve caf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{d55c}\u{ad6d}\u{c5b4} \u{1f600} \u{627}\u{644}\u{639}\u{631}\u{628}\u{64a}\u{629} \u{ff5c}",
];

fn asset(path: PathBuf) -> Asset {
    Asset {
        range: None,
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
        path,
    }
}

/// Loads the checkpoint tokenizer that `variable` names with the prepared template of its recipe.
fn load(variable: &str, recipe: &str, name: &str) -> Frontend {
    let root = std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{variable} is required"));
    Frontend::load(
        FrontendAssets {
            tokenizer: asset(root.join("tokenizer.json")),
            tokens: support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap(),
            chat_template: Some(asset(template_fixtures::path(recipe, name))),
        },
        Default::default(),
    )
    .unwrap()
}

/// Request shapes that carry `word` in each slot of the request that holds text: message content
/// and parts, system text, tool schemas, tool-call argument names and values, tool results,
/// reasoning, documents, image captions, and a continued assistant message. Labels starting with
/// `continue` resume the final assistant message; a template may branch on the content it
/// continues.
fn shapes(word: &str) -> Vec<(&'static str, Vec<Value>, Map<String, Value>)> {
    let generate = json!({"add_generation_prompt": true});
    let tools = json!([
        {"type": "function", "function": {"name": "lookup", "description": format!("desc {word}"),
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": word}},
                "required": ["query"]}}},
        {"type": "function", "function": {"name": "write", "description": "Write a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
    ]);
    let call = |id: &str, arguments: Value| json!({"id": id, "type": "function",
        "function": {"name": "lookup", "arguments": arguments}});
    // A call whose arguments arrive as a JSON string. The server parses the string before it
    // renders (`arguments` in helios/src/server/protocol/decode.rs) and hands the template the
    // parsed object next to the original string.
    let string_call = |id: &str, source: String| {
        let arguments: Value = serde_json::from_str(&source).unwrap();
        json!({"id": id, "type": "function",
            "function": {"name": "lookup", "arguments": arguments, "arguments_source": source}})
    };
    let user = |content: Value| json!({"role": "user", "content": content});
    let options = |value: Value| value.as_object().unwrap().clone();
    vec![
        ("plain", vec![user(json!(format!("see {word} here")))], options(generate.clone())),
        ("system", vec![json!({"role": "system", "content": word}), user(json!("u"))], options(generate.clone())),
        ("parts", vec![user(json!([{"type": "text", "text": format!("see {word}")}, {"type": "text", "text": " more"}]))],
            options(generate.clone())),
        ("multi", vec![json!({"role": "system", "content": "S"}), user(json!("a")),
            json!({"role": "assistant", "content": format!("b {word}")}), user(json!(format!("c\n\n{word}  \n")))],
            options(generate.clone())),
        ("tools", vec![user(json!("Find docs"))], options(json!({"add_generation_prompt": true, "tools": tools}))),
        ("tool_call_keys_object", vec![user(json!("Find docs")),
            json!({"role": "assistant", "content": "", "tool_calls": [
                call("c1", Value::Object(Map::from_iter([(word.to_owned(), json!("x"))])))]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "r1"})],
            options(json!({"add_generation_prompt": true, "tools": tools}))),
        ("tool_call_keys_string", vec![user(json!("Find docs")),
            json!({"role": "assistant", "content": "", "tool_calls": [
                string_call("c1", serde_json::to_string(&json!({word: "x"})).unwrap())]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "r1"})],
            options(json!({"add_generation_prompt": true, "tools": tools}))),
        ("tool_call", vec![user(json!("Find docs")),
            json!({"role": "assistant", "content": "", "tool_calls": [call("c1", json!({"query": word}))]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": format!("file has {word}")}), user(json!("thanks"))],
            options(json!({"add_generation_prompt": true, "tools": tools}))),
        ("parallel_calls", vec![user(json!("u")),
            json!({"role": "assistant", "content": null, "tool_calls": [call("c1", json!({"query": "a"})),
                call("c2", json!({"query": format!("b {word}"), "limit": [1, {"k": word}]}))]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "r1"}),
            json!({"role": "tool", "tool_call_id": "c2", "content": word})],
            options(json!({"add_generation_prompt": true, "tools": tools}))),
        ("reasoning", vec![user(json!("u")),
            json!({"role": "assistant", "content": "answer", "reasoning_content": format!("r {word}")}), user(json!("again"))],
            options(json!({"add_generation_prompt": true, "enable_thinking": true}))),
        ("thinking_off", vec![user(json!(format!("u {word}")))],
            options(json!({"add_generation_prompt": true, "enable_thinking": false}))),
        ("documents", vec![user(json!("summarize"))],
            options(json!({"add_generation_prompt": true, "documents": [{"title": word, "text": format!("body {word}")}]}))),
        ("image", vec![user(json!([{"type": "text", "text": format!("see {word}")},
            {"type": "image_url", "image_url": {"url": "u1"}}]))], options(generate.clone())),
        ("two_images", vec![user(json!([{"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "text", "text": word}, {"type": "image_url", "image_url": {"url": "u2"}}]))],
            options(generate)),
        ("continue", vec![user(json!("u")), json!({"role": "assistant", "content": format!("so {word}")})],
            options(json!({"continue_final_message": true}))),
        ("continue_reasoning", vec![user(json!("u")),
            json!({"role": "assistant", "content": "", "reasoning_content": format!("so {word}")})],
            options(json!({"continue_final_message": "reasoning_content"}))),
    ]
}

/// Ordinary text encodes the same with and without the added vocabulary.
fn added_vocabulary_does_not_change_text_without_controls(variable: &str, frontend: &Frontend) {
    let added = frontend.tokenizer().get_added_vocabulary().get_vocab().keys().cloned().collect::<Vec<_>>();
    for text in TEXT {
        let spelled: Vec<_> = added.iter().filter(|token| text.contains(token.as_str())).collect();
        assert_eq!(
            frontend.encode_ordinary(text).unwrap(),
            frontend.encode(text, false).unwrap(),
            "{variable}: text spells added tokens {spelled:?}: {text:?}"
        );
    }
}

/// Request text that spells an added token that is neither special nor named by the chat template
/// is that token, in any amount and next to text. Every marker the contract declares must be
/// special or named, so none of them is among those tokens.
fn lexical_added_tokens_stay_tokens_in_request_text(
    variable: &str,
    source: &str,
    declared: &[u32],
    frontend: &Frontend,
) {
    let lexical: Vec<_> = frontend
        .tokenizer()
        .get_added_vocabulary()
        .get_added_tokens_decoder()
        .iter()
        .filter(|(_, token)| !token.special && !source.contains(&token.content))
        .map(|(id, token)| (*id, token.content.clone()))
        .collect();
    let declared_lexical: Vec<_> = lexical.iter().filter(|(id, _)| declared.contains(id)).collect();
    assert!(
        declared_lexical.is_empty(),
        "{variable}: declared markers that are neither special nor named by the template: {declared_lexical:?}"
    );
    // A template may reject a lone system or assistant message; the user message must render.
    let render = |role: &str, text: &str| {
        let result = frontend.render_chat(
            &[json!({"role": role, "content": text})],
            json!({"add_generation_prompt": true}).as_object().unwrap(),
        );
        assert!(role != "user" || result.is_ok(), "{variable}: {:?}", result.as_ref().err());
        result.ok()
    };
    for (_, token) in &lexical {
        let text = format!("a {token} b{token}c\n{token}{token} <think> {token}");
        let id = frontend.encode(token, false).unwrap();
        assert_eq!(id.len(), 1, "{variable}: {token:?} is not one token");
        for role in ["user", "system", "tool", "assistant"] {
            let Some(rendered) = render(role, &text) else { continue };
            // Every spelling of the token in the prompt is the token.
            let first_pass = frontend.encode(&rendered.text, false).unwrap();
            assert_eq!(
                rendered.token_ids.iter().filter(|t| **t == id[0]).count(),
                first_pass.iter().filter(|t| **t == id[0]).count(),
                "{variable}: {role}: {token:?}"
            );
            // Without a spelled control the whole prompt is the first-pass encoding.
            let plain = render(role, &text.replace("<think> ", "")).unwrap();
            assert_eq!(
                plain.token_ids,
                frontend.encode(&plain.text, false).unwrap(),
                "{variable}: {role}: {token:?}"
            );
        }
    }
}

/// A request that spells no control renders the first-pass encoding of its text: the added
/// vocabulary changes nothing for any request shape. Returns the slots whose request the template
/// rejects.
fn requests_that_spell_no_control_encode_like_their_text(
    variable: &str,
    frontend: &Frontend,
) -> BTreeSet<&'static str> {
    let mut rejected = BTreeSet::new();
    for (label, messages, options) in shapes("plain words") {
        let Ok(rendered) = frontend.render_chat(&messages, &options) else {
            rejected.insert(label);
            continue;
        };
        assert_eq!(
            rendered.token_ids,
            frontend.encode(&rendered.text, false).unwrap(),
            "{variable}: {label}"
        );
    }
    rejected
}

/// What the comparison of a request slot did with the spellings: how many it compared, how many
/// the template rejected, and which it rendered with different bytes around the spelled word.
#[derive(Debug, Default)]
struct Slot {
    compared: usize,
    rejected: usize,
    branched: Vec<&'static str>,
}

/// Whether `spelled` is `baseline` with each `ZZZ` replaced by `spelling`, written as is or
/// escaped as inside a JSON string (a template serializes tool schemas and arguments).
fn differs_in_the_word_alone(baseline: &str, spelled: &str, spelling: &str) -> bool {
    let escaped = serde_json::to_string(spelling).unwrap();
    let escaped = &escaped[1..escaped.len() - 1];
    let mut parts = baseline.split("ZZZ");
    let Some(mut rest) = spelled.strip_prefix(parts.next().unwrap()) else { return false };
    for part in parts {
        let Some(after) = [spelling, escaped].iter().find_map(|word| rest.strip_prefix(word)) else {
            return false;
        };
        let Some(after) = after.strip_prefix(part) else { return false };
        rest = after;
    }
    rest.is_empty()
}

/// Request text that spells a control does not add or remove a control the template writes: for
/// every added token, its count in the prompt equals the count with a neutral word in the same
/// slot. That includes the spelled token itself, which must not become a token, unless it is
/// lexical (neither special nor named by the template), when it adds one token per spelling.
/// A spelling is skipped when the template rejects the request or writes different bytes around
/// the two words; the returned slots count each outcome, so a caller can fail a slot that no
/// longer compares. A continued assistant message is not compared: a template may branch on the
/// content it continues.
fn spelling_a_control_leaves_the_templates_controls_unchanged(
    variable: &str,
    source: Option<&str>,
    frontend: &Frontend,
) -> BTreeMap<&'static str, Slot> {
    const SPELLINGS: &[&str] = &[
        "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "<tool_response>",
        "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|system|>",
        "<|image|>", "<image>", "<|image_pad|>", "<|vision_start|>", "<|endoftext|>", "[gMASK]",
        "<sop>", "<|turn>", "<turn|>", "<|channel>", "<|\"|>", "<\u{ff5c}User\u{ff5c}>",
        "<\u{ff5c}Assistant\u{ff5c}>", "\u{ff5c}DSML\u{ff5c}",
    ];
    let vocabulary = frontend.tokenizer().get_added_vocabulary();
    let added: Vec<u32> = vocabulary.get_added_tokens_decoder().keys().copied().collect();
    let count = |ids: &[u32], token: u32| ids.iter().filter(|id| **id == token).count();
    let neutral = shapes("ZZZ");
    let mut slots: BTreeMap<&'static str, Slot> = BTreeMap::new();
    for spelling in SPELLINGS {
        let own = vocabulary.get_vocab().get(*spelling).copied();
        // A native encoder tokenizes all request text as ordinary text: nothing is lexical.
        let lexical = source.is_some_and(|source| {
            own.is_some_and(|id| {
                !vocabulary.get_added_tokens_decoder()[&id].special && !source.contains(spelling)
            })
        });
        for ((label, messages, options), (_, baseline_messages, baseline_options)) in
            shapes(spelling).iter().zip(&neutral)
        {
            if label.starts_with("continue") {
                continue;
            }
            let slot = slots.entry(label).or_default();
            let spelled = frontend.render_chat(messages, options);
            let baseline = frontend.render_chat(baseline_messages, baseline_options);
            let (Ok(spelled), Ok(baseline)) = (spelled, baseline) else {
                slot.rejected += 1;
                continue;
            };
            // A template may branch on request text (Nemotron skips its empty reasoning
            // block for an assistant message that spells `<think>`); the counts are
            // comparable only when the two prompts differ in the word alone.
            if !differs_in_the_word_alone(&baseline.text, &spelled.text, spelling) {
                slot.branched.push(spelling);
                continue;
            }
            slot.compared += 1;
            assert_eq!(
                frontend.decode(&spelled.token_ids, false).unwrap(),
                spelled.text,
                "{variable}: {label}: {spelling}"
            );
            for &id in &added {
                let (spelled, baseline) = (count(&spelled.token_ids, id), count(&baseline.token_ids, id));
                if lexical && Some(id) == own {
                    assert!(spelled >= baseline, "{variable}: {label}: {spelling} loses its token");
                } else {
                    assert_eq!(spelled, baseline, "{variable}: {label}: {spelling} changes token {id}");
                }
            }
            assert_eq!(spelled.image_sources.len(), baseline.image_sources.len(), "{variable}: {label}");
        }
    }
    slots
}

/// Fails when a slot or spelling outside the checkpoint's lists is skipped. `rejects` lists the
/// slots that the checkpoint's template is expected to reject, and `branches` the (slot,
/// spelling) pairs for which it writes different bytes around the spelled word (it branches on
/// the request text). Every other slot compares every spelling; an entry that no longer applies
/// must leave its list.
fn slots_that_stop_comparing_fail(
    variable: &str,
    expected: (&[&str], &[(&str, &str)]),
    rejected_by_plain_render: &BTreeSet<&str>,
    slots: &BTreeMap<&'static str, Slot>,
) {
    eprintln!("{variable}: slots {slots:?}");
    let (rejects, branches) = expected;
    let rejects: BTreeSet<_> = rejects.iter().copied().collect();
    assert_eq!(
        *rejected_by_plain_render, rejects,
        "{variable}: slots whose request the template rejects, against the checkpoint's list"
    );
    let uncompared: BTreeSet<_> = slots
        .iter()
        .filter(|(_, slot)| slot.compared == 0)
        .map(|(label, _)| *label)
        .collect();
    let rejects: BTreeSet<_> = rejects.into_iter().filter(|label| !label.starts_with("continue")).collect();
    assert_eq!(uncompared, rejects, "{variable}: slots that never compare, against the checkpoint's list");
    let branched: BTreeSet<_> = slots
        .iter()
        .flat_map(|(label, slot)| slot.branched.iter().map(move |spelling| (*label, *spelling)))
        .collect();
    assert_eq!(
        branched,
        branches.iter().copied().collect(),
        "{variable}: (slot, spelling) pairs the template renders differently, against the checkpoint's list"
    );
}

/// `contract` names the recipe's own frontend contract fixture. Its declared markers are controls
/// too, as the server loads them, and each must be special or named by the template. `expected`
/// lists what the checkpoint's template is expected to skip (see `slots_that_stop_comparing_fail`).
fn check(variable: &str, recipe: &str, name: &str, contract: &str, expected: (&[&str], &[(&str, &str)])) {
    let frontend = load(variable, recipe, name);
    let declared = model::declared_control_tokens(&frontend, &support::frontend_contract(contract)).unwrap();
    assert!(!declared.is_empty(), "{variable}: the contract declares no markers");
    let frontend = frontend.with_controls(declared.clone());
    let source = std::fs::read_to_string(template_fixtures::path(recipe, name)).unwrap();
    added_vocabulary_does_not_change_text_without_controls(variable, &frontend);
    lexical_added_tokens_stay_tokens_in_request_text(variable, &source, &declared, &frontend);
    let rejected = requests_that_spell_no_control_encode_like_their_text(variable, &frontend);
    let slots = spelling_a_control_leaves_the_templates_controls_unchanged(variable, Some(&source), &frontend);
    slots_that_stop_comparing_fail(variable, expected, &rejected, &slots);
}

/// DeepSeek V4.1 has no chat template: its native encoder writes the prompt's controls and
/// tokenizes all request text as ordinary text, so no added token is lexical and no declared
/// marker needs to be special or named.
fn check_native(variable: &str, contract: &str, expected: (&[&str], &[(&str, &str)])) {
    let root = std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{variable} is required"));
    let tokens = support::tokens(&asset(root.join("tokenizer_config.json"))).unwrap();
    let frontend = Frontend::load_with_settings(
        FrontendAssets { tokenizer: asset(root.join("tokenizer.json")), tokens: tokens.clone(), chat_template: None },
        Default::default(),
        &frontend::FrontendSettings {
            tokenizer: tokens,
            eos_token_ids: vec![],
            prompt_encoder: frontend::PromptEncoder::DeepseekV41,
            image_placeholder_token_id: None,
            template_encoding: Default::default(),
            image: None,
        },
    )
    .unwrap();
    let declared = model::declared_control_tokens(&frontend, &support::frontend_contract(contract)).unwrap();
    let frontend = frontend.with_controls(declared);
    added_vocabulary_does_not_change_text_without_controls(variable, &frontend);
    let rejected = requests_that_spell_no_control_encode_like_their_text(variable, &frontend);
    let slots = spelling_a_control_leaves_the_templates_controls_unchanged(variable, None, &frontend);
    slots_that_stop_comparing_fail(variable, expected, &rejected, &slots);
}

#[test]
#[ignore = "requires FLARE_GLM_TEST_SNAPSHOT with the GLM-5.3 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn glm53_ordinary_tokens() {
    check("FLARE_GLM_TEST_SNAPSHOT", "glm53", "chat_template", "glm53", (&[], &[("multi", "</think>")]));
}

#[test]
#[ignore = "requires FLARE_TEST_SNAPSHOT with the Qwen3.8 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn qwen38_ordinary_tokens() {
    check("FLARE_TEST_SNAPSHOT", "qwen38", "chat_template", "qwen38", (&[], &[]));
}

#[test]
#[ignore = "requires FLARE_STEP5_TEST_SNAPSHOT with the Step-5 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn step5_ordinary_tokens() {
    check("FLARE_STEP5_TEST_SNAPSHOT", "step5", "chat_template", "step5", (&[], &[("multi", "</think>")]));
}

#[test]
#[ignore = "requires FLARE_NEMOTRON_TEST_SNAPSHOT with the Nemotron-3.5 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn nemotron_ordinary_tokens() {
    check("FLARE_NEMOTRON_TEST_SNAPSHOT", "nemotron_h", "chat_template", "nemotron_h",
        (&[], &[("image", "<image>"), ("multi", "<think>"), ("multi", "</think>"), ("two_images", "<image>")]));
}

#[test]
#[ignore = "requires FLARE_MIMO_TEST_SNAPSHOT with the MiMo-V2.6 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn mimo_ordinary_tokens() {
    check("FLARE_MIMO_TEST_SNAPSHOT", "mimo_v2", "chat_template", "mimo_v2", (&[], &[]));
}

#[test]
#[ignore = "requires FLARE_GEMMA4_TEST_SNAPSHOT with the Gemma 4 tokenizer and FLARE_TEST_TEMPLATE_DIR"]
fn gemma4_ordinary_tokens() {
    check("FLARE_GEMMA4_TEST_SNAPSHOT", "gemma4", "chat_template.google-e2b-e4b", "gemma4",
        (&[], &[("multi", "<|channel>")]));
}

#[test]
#[ignore = "requires FLARE_DEEPSEEK_TEST_SNAPSHOT with the DeepSeek V4.1 tokenizer"]
fn deepseek_ordinary_tokens() {
    // The native encoder rejects documents, images and continuation.
    let rejects = ["continue", "continue_reasoning", "documents", "image", "two_images"];
    check_native("FLARE_DEEPSEEK_TEST_SNAPSHOT", "deepseek41", (&rejects, &[]));
}
