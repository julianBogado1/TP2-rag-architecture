from app.core.llm_client import LLMClient
from app.models.prompt_score import PromptScore


_SYSTEM_PROMPT = """You convert a user's natural-language music request into a
structured score. Set semantic_query to a clean English rephrasing optimized for
embedding search: strip emojis, fix Spanglish, expand abbreviations.

NULL VS 0.0 (critical):
For every mood/wants_* field and every audio_features axis, output a float in
[0, 1] ONLY when the prompt actually implies that dimension. If the prompt is
silent on a dimension, leave the field NULL (omit it). Reserve 0.0 for prompts
that EXPLICITLY negate the dimension ("no high energy", "nothing sad",
"not popular"). "Silent" and "explicitly absent" are different states — keep
them distinguishable.

AUDIO FEATURES (critical):
audio_features describes the SOUND the user wants, inferred from mood, vibe, or
activity words — NOT only from explicit numbers. A mood/vibe/activity DOES imply
audio axes, so infer them (this is "implied", not "silent"). Map cues to axes
(each in [0,1]); leave an axis NULL only when the prompt gives no cue for it:
  happy / upbeat / feel-good                  -> high valence, high energy
  sad / melancholic / heartbreak / down       -> low valence, low energy
  calm / relaxing / chill / focus / study / sleep -> low energy, high acousticness, low danceability
  energetic / hype / workout / gym / running  -> high energy, high tempo_norm
  party / dance / club                        -> high danceability, high energy
  acoustic / unplugged / stripped-down        -> high acousticness
  instrumental / no vocals / background        -> high instrumentalness
Use values near the extremes for strong cues (e.g. "very calm" -> energy ~0.1) and
mid values for mild ones. Combine cues when several apply. Leave audio_features
axes NULL only when the prompt is purely about genre/artist/language with no
mood, vibe, or activity cue at all.

WANTED VS UNWANTED (critical):
Genre, artist, and song mentions carry polarity. Route them correctly:
- wanted_genres / wanted_artists: positive mentions ("I want rap",
  "songs by Taylor Swift", "artists like Ed Sheeran" -> wanted_artists=["Ed Sheeran"]).
- unwanted_genres / unwanted_artists / unwanted_songs: negative mentions.
  Negation cues: "no", "not", "without", "avoid", "except", "don't want",
  "nothing", "hate", "skip". Example: "don't give me any rap songs" ->
  unwanted_genres=["rap"]. A negated mention NEVER goes into a wanted_* list.
- Leave any list null when no mention of that kind appears.

GENRE VOCABULARY (critical):
wanted_genres and unwanted_genres accept ONLY these canonical values:
"country", "misc", "pop", "rap", "rb", "rock".
Map common terms to the canonical form when possible:
  hip-hop / hip hop / trap            -> "rap"
  r&b / rnb / soul                    -> "rb"
  metal / punk / alternative / indie  -> "rock"
  electronic / edm / house / techno   -> "misc"
  folk / classical / jazz / latin     -> "misc"
If a mentioned genre does not map to one of these six, OMIT it (do not invent
or substitute). Empty list -> null.

LANGUAGE RULES (critical):
- preferred_language must be a 2-letter ISO 639-1 code: "es", "en", "pt", "fr",
  "it", "de", "ja", "ko", etc. NEVER the full name ("Spanish", "English").
- ONLY set preferred_language if the user EXPLICITLY asks for a language
  ("canciones en español", "songs in English", "musique française"). Writing the
  prompt itself in Spanish is NOT a request for Spanish-language songs — leave
  preferred_language null in that case.

Never invent data the user didn't imply."""


class PromptParserService:
    """Calls the LLM with a structured-output schema to turn raw prompt → PromptScore."""

    def __init__(self, llm_client: LLMClient, model: str) -> None:
        self._llm = llm_client
        self._model = model

    def parse(self, raw_prompt: str) -> PromptScore:
        return self._llm.parse_structured(
            model=self._model,
            system=_SYSTEM_PROMPT,
            user=raw_prompt,
            schema=PromptScore,
        )
