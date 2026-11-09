Faithfully and fully transcribe everything said in this recording.

Recording context:
{chunk_context}

Write the complete transcript in English. Keep speech originally in English in
English. Faithfully translate every non-English turn into English. When a turn
uses more than one language, produce one coherent English rendering of that
turn without summarizing, interpreting, or editorializing.

Preserve the exact chronological order of all speech. Return one JSON object
with exactly one key, `turns`. Its value must be a nonempty array of objects,
each with exactly these string keys: `timestamp`, `speaker`, and `content`.

`timestamp` must be the turn start in exactly `HH:MM:SS` form, including two
digits for every component. `speaker` must be a stable numbered label such as
`Speaker 1` unless a specific identity is reliably evident. `content` is the
complete English rendering of that turn.

Use stable numbered speaker labels such as Speaker 1, Speaker 2, and Speaker 3
by default. Use a specific speaker identity only when it is reliably evident
from the recording. Identify a speaker change whenever possible.

Transcribe all speech completely. Do not omit material, summarize, clean up
grammar, interpret meaning, or add information. When a word or phrase is not
clearly intelligible but can be guessed with confidence, transcribe the guess
in square brackets, for example [project timeline]. When speech is wholly
unintelligible, write [inaudible].

Return only the JSON object. Do not add a title, heading, comment, explanation,
summary, conclusion, preface, closing text, Markdown, or any fields other than
the required `turns` array and its turn objects.
