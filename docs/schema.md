# Response configuration schema

`responses.yml` maps incoming chat requests to the text the simulator returns. It controls **what** the model says, never **when**: token timing comes from the simulator and the hardware configuration.

Version: **1**

## Overview

```yaml
version: 1
seed: 42

responses:
  - id: greeting
    match: { contains: ["hello", "hi there"] }
    content: "Hello! How can I help you today?"

  - id: weather
    match: { regex: "weather in (?P<city>\\w+)" }
    variants:
      - "It's sunny in {city} today."
      - "Expect light rain in {city}."
      - "{city} is cloudy with a chance of storms."

  - id: echo
    match: { starts_with: "/echo" }
    content: "You said: {last_user_message}"

  - id: long-output
    match: { starts_with: "/long" }
    variants:
      - synthetic: { tokens: [500, 2000] }
      - "Short answer this time."

default:
  variants:
    - "I'm a simulated model."
    - "This is a canned response from llm-simulator."
```

Each request is checked against `responses` from top to bottom. The first entry that matches produces the response. If nothing matches, `default` is used.

## Top level

| Key | Type | Required | Default | Description |
|---|---|---|---|---|
| `version` | int | yes | — | Schema version. Must be `1`. |
| `seed` | int | no | `0` | Default seed for variant selection. A request's `seed` parameter overrides it. |
| `responses` | list of [Entry](#entry) | no | `[]` | Response rules, evaluated in order. The first match wins. |
| `default` | [Body](#body) | yes | — | Response used when no entry matches. |

## Entry

| Key | Type | Required | Description |
|---|---|---|---|
| `id` | str | yes | Unique identifier matching `^[a-z0-9][a-z0-9_-]*$`. `default` is reserved. Used as a label in logs and metrics. |
| `match` | [Match](#match) | yes | Condition the request must satisfy. |
| `content` / `variants` / `synthetic` | [Body](#body) | yes | The response, with the body keys written inline in the entry. |

## Match

A match has exactly one operator. All matching is case-insensitive.

| Key | Type | Matches when |
|---|---|---|
| `exact` | str or list[str] | the input equals the value |
| `contains` | str or list[str] | the value is a substring of the input |
| `starts_with` | str or list[str] | the input starts with the value |
| `regex` | str or list[str] | `re.search` with `re.IGNORECASE` finds the pattern in the input |

- **Input** is the `content` of the last message with `role: user`, with leading and trailing whitespace removed, so `"hello\n"` matches `exact: "hello"`. If there is no user message, the input is the empty string.
- **Lists** are OR: the entry matches if any value matches.
- **Regex named groups** (`(?P<name>...)`) become [template variables](#template). When `regex` is a list, a placeholder may only refer to groups defined in every pattern.

## Body

A body has exactly one of these keys:

| Key | Type | Description |
|---|---|---|
| `content` | [Template](#template) | A single fixed response. |
| `variants` | list of [Variant](#variant), at least 1 | One variant is selected deterministically, each equally likely (see [Evaluation](#evaluation)). |
| `synthetic` | [Synthetic](#synthetic) | Generated filler text of a given length. |

## Variant

A variant is either a plain string, which is shorthand for `{ content: "<string>" }`, or an object with exactly one of:

| Key | Type | Description |
|---|---|---|
| `content` | [Template](#template) | A fixed response. |
| `synthetic` | [Synthetic](#synthetic) | Generated filler text. |

Variants cannot be nested.

## Synthetic

| Key | Type | Description |
|---|---|---|
| `tokens` | int ≥ 1, or `[min, max]` with `1 ≤ min ≤ max` | Response length in tokens, counted by the configured tokenizer. A range is resolved deterministically per request. |

Synthetic text comes from a fixed lorem-ipsum corpus, repeated as many times as the length needs. Use it when only the response length matters, for example in load tests.

## Template

A template is a string with `{name}` placeholders. Write `{{` and `}}` to get literal braces.

| Variable | Value | Available in |
|---|---|---|
| `last_user_message` | The content of the last user message, exactly as sent (whitespace not removed) | everywhere |
| `model` | The requested model name | everywhere |
| `<group>` | A named regex group from the entry's `match`. An optional group that did not participate renders as `""`. | entries with a `regex` match |

## Evaluation

1. **Input.** Take the content of the last `user` message and remove leading and trailing whitespace. A message's content may be a string or an array of text parts (`[{"type": "text", "text": "..."}]`); parts are joined with no separator, so `["Summarise:", " it"]` is the input `Summarise: it` and `{last_user_message}` renders it the same way. OpenAI does not document how it joins parts, so this is an assumption to revisit if measurement shows otherwise; the separator is `PART_SEPARATOR` in `client/convert.py`. Other part types (images, audio, files) are rejected with a 400. This trimmed input is used for matching and for variant selection.
2. **Match.** Walk `responses` in file order. The first matching entry wins. If none match, use `default`.
3. **Select a variant.**
   - `digest = sha256(f"{seed}\0{entry_id}\0{input}")` over the UTF-8 bytes, where `seed` is the request's `seed` if it was given, otherwise the file's `seed`, written in decimal.
   - `h` is bytes 0–7 of the digest, read as a big-endian unsigned integer.
   - Choose the variant at index `h mod len(variants)`.
4. **Render.** Substitute template variables, or generate synthetic text. For a `[min, max]` range, the length is `min + (h' mod (max − min + 1))`, where `h'` is bytes 8–15 of the same digest, read the same way. Using separate bytes keeps the variant and the length independent.
5. **Tokenize.** The generator splits the rendered text into tokens and truncates at the request's `max_tokens`. When it truncates, `finish_reason` is `length`; otherwise it is `stop`.

The same request with the same seed always produces the same response, including across process restarts. This is why selection uses `sha256` instead of Python's built-in `hash()`, which is randomized per process.

## Validation

The file is validated once at startup. Any of the following stops the server from starting with a descriptive error:

- `version` is missing or not `1`, or `default` is missing.
- Any object contains an unknown key.
- An entry `id` is invalid or duplicated.
- A `match` has no operator or more than one.
- An operator has an empty list or an empty string.
- A `regex` fails to compile.
- A body or variant has no content key, or more than one.
- `variants` is empty.
- A synthetic range has `min > max` or `min < 1`.
- A template uses a placeholder that is not available in its context.
- A regex named group is called `last_user_message` or `model`, which would clash with a built-in variable.

## Not supported in v1

- Boolean combinators (AND/NOT), conditionals or expressions.
- Multi-turn or stateful scripts.
- Timing, delays or latency of any kind. These belong to the simulator.
- Error injection, such as returning a 429 or 500 response.
