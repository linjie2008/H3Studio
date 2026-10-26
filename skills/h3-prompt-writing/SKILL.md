---
name: h3-prompt-writing
description: Write MiniMax H3 video generation prompts for T2VA, I2VA, FL2VA, L2VA, and Ref2VA following the official prompt structure (instruction line + integrated_multimodal_description / overall_soundscape / non_diegetic_music). Use when rewriting multimodal requests into H3 prompt structures, composing the three core fields, aligning keyframes, defining reference labels, writing camera motion in the official vocabulary, or formatting dialogue with H3 speech blocks.
metadata:
  compatibility: Portable to any agent that can read local files — no external API calls, MiniMax Hub tools, or proprietary runtime required. The agents/openai.yaml file only adds optional ChatGPT/Codex UI metadata; it does not restrict the skill to OpenAI agents.
---

# H3 Prompt Writing

Official structure first, local calibration second. Sections marked **[OFFICIAL]** come from `references/base-en.txt` / `references/ref-en.txt` and override personal style habits. The **[LOCAL]** section is measured on this workstation and only tightens numbers — it never contradicts the official format.

## Mode Abbreviations

Official docs use these abbreviations without ever spelling them out; the expansions below are derived from usage and mapped to the workbench's internal mode names. **VA = Video + Audio**, meaning the mode generates a synchronized soundtrack natively (speech, ambience, score in one pass) — not picture first, dub later.

| Abbr. | Expansion | Meaning | Input | Workbench mode |
|---|---|---|---|---|
| **T2VA** | Text to Video with Audio | 文生视频（带声音） | text only | `t2v` |
| **I2VA** | Image to Video with Audio | 首帧图生视频（带声音） | first frame | `i2v` |
| **FL2VA** | First-and-Last-frame to Video with Audio | 首尾帧生视频（带声音） | first + last frame | `fl2v` |
| **L2VA** | Last-frame to Video with Audio | 末帧生视频（带声音） | last frame | `l2v` |
| **Ref2VA** | Reference to Video with Audio | 全参考生视频（带声音） | image / video / audio mix | `r2v`, `v2v`, `rv2v` |

Full-reference sub-modes: `r2v` = pure reference generation, `v2v` = video editing, `rv2v` = video editing with reference assets.

## Two Usage Contexts — Do Not Mix Them

This skill serves two very different jobs. Choose the context first; mixing them is the single most common way to break a run.

**A. Workbench shot fields (the normal case).** The engine already assembles the whole official prompt: for `t2v/i2v/fl2v/l2v` it wraps your text into `integrated_multimodal_description` + `overall_soundscape` + `non_diegetic_music`; for `r2v/v2v/rv2v` it builds the full six-section Ref2VA prompt (subject definitions, summary, retention analysis, detailed description, soundscape, music) from the shot fields and the reference assets.

In context A you write **only** the pieces that land inside `detailed_description`:
- one or two sentences of shot description (composition + subject + action + camera)
- the reference tag, e.g. `画面以 <Picture 3> 为起幅`
- dialogue already supplied through structured fields stays in `dialogue_text`, `dialogue_voiceover`, `dialogue_time`, `speaker_description`; inline narration supplied in the prompt stays inline once, as described below
- visible text through `on_screen_text`
- ambience and score through `overall_soundscape` / `non_diegetic_music` fields

**Never paste section headers into the prompt box.** Writing `integrated_multimodal_description:`, `subject_definitions:`, `summary:`, `retention_analysis:`, `detailed_description:`, `overall_soundscape:` or `non_diegetic_music:` inside a shot prompt makes the engine nest a whole prompt inside its own field — duplicated instructions, contradictory structure, and the model starts reading the headers aloud.

**B. Hand-written full prompt (rewrite / external H3 call).** Here you have no engine scaffolding, so you write the complete official structure yourself: instruction line + the three fields for base modes, or the six sections for Ref2VA. Only in this context do the `[OFFICIAL]` templates below get typed out literally.

| | A. Workbench shot | B. Hand-written prompt |
|---|---|---|
| Structure | engine builds it | you write it |
| Body length | 1–2 sentences | 350–500 words (Ref2VA generation) |
| Section headers | never include | required |
| Description language | English preferred; a short Chinese camera sentence (≤25 chars) is acceptable and tested clean | English |
| Dialogue | existing `dialogue_text` field, or one inline `<d>` block for narration authored in the prompt | `<d>[Chinese] …</d>` inside the description |

### [LOCAL] Workbench narration optimization contract

Preserve the spoken words, names and language, not multiple copies of their original formatting. The optimizer returns description text and cannot move inline speech into another field: retain that speech once in `<d>[language] original words</d>`, with speaker identity and off-screen delivery outside the block. Speech already supplied in a separate field stays there; do not copy it into the description.

A quoted note such as `旁白“彪投股份，专业成就未来”` and `<d>[Chinese] 彪投股份，专业成就未来。</d>` represent the same utterance despite terminal punctuation. Consolidate them into one inline block. Do not append the original note, transcript, summary or reminder after the description. Preserve explicitly requested repeats at different times or by different speakers.

Voiceover describes the sound source; it does not require removing hands or people from the user's scene. Do not invent a visible speaker, lip movement, speech speed, or a requirement for a short line to fill the entire clip. Before returning an optimized description, check that every intended utterance appears once with its words unchanged.

## Workflow

1. Identify the input mode: T2VA, I2VA, FL2VA, L2VA, or full-reference Ref2VA.
2. Base/keyframe modes → read `references/base-en.txt`, follow its final prompt structure exactly.
3. Full-reference mode → read `references/ref-en.txt`, follow its six-section rewrite format.
4. Preserve field names, section order, labels, and timing notation verbatim.
5. Run the pre-flight checklist at the bottom before submitting.

## Base Modes

- T2VA: build the complete audiovisual timeline from text.
- I2VA: T2VA body + first-frame instruction + a visual path that develops forward from the first frame.
- FL2VA: T2VA body + first-and-last-frame instruction + a continuous path between the two frames.
- L2VA: T2VA body + last-frame instruction + a path that converges onto the last frame.

## [OFFICIAL] Final Prompt Structure

**Instruction line (I2VA / FL2VA / L2VA only — always the first line, followed by ONE blank line).** T2VA has no instruction line and starts directly with the three fields.

```text
I2VA:  For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
FL2VA: How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
L2VA:  How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
```

`N` = index of the actual final shot; `S.SS` = effective duration with exactly two decimals.

**Then exactly three fields, in this order:**

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

- `integrated_multimodal_description` — visuals, actions, shots, speakers, dialogue, diegetic audio along the timeline.
- `overall_soundscape` — ambient + physical action sounds + non-verbal human sounds across the whole video.
- `non_diegetic_music` — music only the audience hears.

## [OFFICIAL] Write the Description in English

*Applies to context B (hand-written prompts). In context A the engine supplies the structure and a short Chinese camera sentence is acceptable.*

The description body is English. Only these stay verbatim in the original language: dialogue inside `<d>`, lyrics, and on-screen text inside `""`.

Start `[Shot 1]` with style + initial composition: `[Shot 1] Live-action, cinematic, a high aerial shot frames a lakeside campus at dusk.` Style words: `Cinematic`, `live-action`, `2D-animated`, `3D CG`, `claymation`, `watercolor`, `vintage film`. For keyframe modes derive the style from the reference image.

Everything described must be visible or audible: style, composition, subject appearance/position, scene and props, actions and reactions, camera, speech, synchronized sound.

## [OFFICIAL] Shots and Cuts

- No timestamp on `[Shot 1]`.
- Later shots begin with a strictly increasing cut time inside the duration: `[Shot 2] At 00:03.500, the camera cuts to ...`
- Cut verbs: `the camera cuts to`, `the shot cuts to`, `the shot transitions to`, `the shot changes to`, `the shot switches to`. Cross-dissolve / fade / wipe only when explicitly requested.
- A cut must introduce new information (subject, space, state, viewpoint, time). For a small distance or angle change, use camera motion instead.

## [OFFICIAL] Camera Motion = Type + Amplitude + Speed

Write it as natural action inside the sentence, never as labels stacked at the end.

| Type | Meaning | Type | Meaning |
|---|---|---|---|
| Zoom In / Out | focal length changes, body still | Pan Left / Right | lens pivots horizontally |
| Push In / Pull Out | camera moves forward / backward | Truck Left / Right | camera translates horizontally |
| Tilt Up / Down | lens pivots vertically | Pedestal Up / Down | whole camera moves up / down |
| Arc Shot | camera arcs around subject | Tracking Shot | camera follows a moving subject |
| Static Shot | position and lens still | Shake Slightly / Strongly | camera shake |
| POV | subject's point of view | Roll Clockwise / Counterclockwise | roll around lens axis |

Amplitude: `with small amplitude` / `with large amplitude`. Speed: `at slow speed` / `at fast speed`. Omit both when medium/normal — that is the default.

```text
The camera pushes in with small amplitude at slow speed toward the entrance sign.
The camera pans right with large amplitude at fast speed, revealing the lake.
```

## [OFFICIAL] Speakers, Dialogue, Voiceover

- Speaking / singing / off-screen voices get stable IDs `(S1)`, `(S2)`; several at once → `(S1,S2)`. The same speaker keeps the same ID across shots. Silent characters get no ID.
- On first appearance give identity information (type, age, gender, on-screen or not, pitch, timbre, rate, accent). Identity phrase, ID, action and delivery stay **outside** `<d>`; inside `<d>` put only the language tag plus the user's exact words — verbatim, untranslated.

```text
The middle-aged owner with a calm, low voice (S1) says: <d>[Chinese] 彪投股份，扎根开原。</d>
The two children (S1,S2) shout together, <d>[Chinese] 等等我们！</d>
```

- Voiceover must use the exact phrase `says in an off-screen voiceover`, and immediately after the `<d>` block state that the on-screen character's lips stay closed:

```text
The narrator (S1) says in an off-screen voiceover: <d>[Chinese] 一湖碧水，几座绿岛。</d> while the on-screen character's lips remain completely closed.
```

- Speech crossing a cut: `<scenetrans>` at both connecting points plus an explicit continuity phrase. Truncated speech: `<cutoff>`.

## [OFFICIAL] On-Screen Text

Anything actually visible on screen (banner, sign, label, subtitle, neon) goes in English double quotes, verbatim, untranslated:

```text
A metal sign reading "彪投股份" is mounted at the main entrance.
```

## [OFFICIAL] overall_soundscape and non_diegetic_music

- `overall_soundscape`: 1–4 English sentences, one continuous paragraph — wind, water, traffic, footsteps, fabric, impacts, breathing, laughter. Never repeat dialogue, singing, or diegetic music here. `N/A` only for requested total silence.
- `non_diegetic_music`: 1–3 English sentences on instrumentation, tempo, rhythm, dynamics. No mood adjectives, no explanation of emotional purpose. `N/A` when there is none.

```text
overall_soundscape: Water laps gently against the lakeshore while wind moves through the trees. Footsteps cross gravel, and a distant gate clicks shut.

non_diegetic_music: Sparse piano notes at a slow tempo, joined by sustained low strings that swell slightly before fading out.
```

## [LOCAL] Speech Budget — measured on this workstation

The official structure says nothing about density, but density is what actually breaks the audio here. Measured failure: 32 Chinese characters forced into a 2.5 s window (12.8 chars/s ≈ 2.5× natural pace) came out as 「至少无知事」 for 「的城市」, with 「彪投股份」 dropped entirely.

1. **≤ 5 characters per second of speech time** (safe 4–5). 2.5 s window → ≤ 12 chars; 5 s window → ≤ 24 chars. Split into two shots rather than exceeding it.
2. **One line per shot.** Never stack two sentences into one window; never repeat a line twice in a prompt.
3. **Proper nouns degrade first** (「彪投股份」→「标头谷分」, 「开原」→「开源」). Shorten to two characters or let on-screen text carry the full name.
4. **Any Chinese prose in the prompt can be spoken aloud.** Measured: the constraint 「整段必须始终是夜晚」 was voiced as 「整段必须是妨碍」. Keep Chinese to the `<d>` dialogue, quoted on-screen text, and image tags; everything else — camera, style, soundscape — stays English.
5. **Never write constraint sentences** (「绝对不要白天」「本段仅此一条旁白」). They get read out and add nothing.
6. **Lock lighting and continuity with the first frame, not with words.** A night first frame stays night; prose cannot force a daylight frame to night.
7. **ASR gate is mandatory.** Transcribe every generated segment. Homophone slips (彪投→彪头) pass; extra sentences, dropped words, or slurred proper nouns mean re-roll. Shortening the line fixes garbling far better than re-rolling at the same length.

## [LOCAL] Reference Tags Bind Images

In this workbench a shot keeps its reference image only when the prompt contains an authored tag such as `<Picture 3>` matching the project's `reference_image_numbers`. A "cleaned up" description without the tag silently drops every reference and the run fails with 「参考素材模式需要至少一个参考文件」. Always keep the tag inside the description (e.g. `the campus shown in <Picture 3>`) and keep `N` consistent with the project's numbering.

### [LOCAL] Identity across continued shots

When a later shot changes people, bind that shot's own person image and keep shared scene or vehicle images in their scene or object roles. A shared scene or vehicle reference is not evidence that the person stayed the same. Check both the project-wide `<Picture N>` tags and the shot-local image slots shown by the workbench; the engine remaps the numbers before sending them to H3.

The previous clip's tail is a visual condition, not merely a text cue. If it still shows the outgoing person's face or clothes, a longer continuation guide can overpower the new person image. Inspect the actual outgoing frames and the media passed to H3; use a short scene-only tail where available, while keeping the camera path, background and motion coherent. If the media conditions are wrong, fix the bindings or handoff in the workbench or runner. Repeating "different person" in prose cannot repair them.

## Full-Reference Mode

Ref2VA rewrites use `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music` in that order. Reference labels stay consistent across all sections. Read `references/ref-en.txt` for label rules, retention analysis, and complete examples.

## Output Rules

- Write rewrite sections in English; preserve dialogue, lyrics, and visible scene text in their original language.
- Describe each shot by composition, subjects, environment, actions, camera, sound, and the exact point where referenced content appears.
- Avoid plot summaries, unresolved reference labels, and timing that doesn't match the requested duration.
- Keep reference labels consistent (`<Picture 1>`, `<Video 1>`, `<Audio 1>`) across every section.
- Match the total duration of the description to the requested video length (4–15 s).
- When using keyframes (I2VA / FL2VA / L2VA), state explicitly how the first and/or last frame connects to the timeline.

## Pre-flight Checklist

- [ ] Mode-specific instruction line present (I2VA / FL2VA / L2VA), then one blank line.
- [ ] Exactly three fields, correct names and order.
- [ ] Description in English; only dialogue, lyrics, on-screen text keep the original language.
- [ ] `[Shot 1]` has no timestamp; later cuts use strictly increasing `At 00:0X.XXX`.
- [ ] Camera motion written as type + amplitude + speed inside a sentence.
- [ ] Speaker IDs stable; `<d>` holds only `[Language]` + exact words; voiceover uses the official phrase and states lips closed.
- [ ] On-screen text verbatim in double quotes.
- [ ] `overall_soundscape` 1–4 sentences without dialogue/music; `non_diegetic_music` 1–3 sentences without mood words, or `N/A`.
- [ ] Every referenced `<Picture N>` exists in the project's numbering.
- [ ] Speech density ≤ 5 chars/second; one line per shot.
- [ ] No Chinese constraint sentences anywhere in the prompt.
