//! Finds stop strings in generated text as it arrives, holding back only what could still turn
//! into a match, so text ahead of a stop string is never delayed by more than that.

use std::borrow::Cow;

pub(crate) struct StopMatcher<'a> {
    stops: &'a [String],
    held: String,
}

impl<'a> StopMatcher<'a> {
    pub(crate) fn new(stops: &'a [String]) -> Self {
        Self {
            stops,
            held: String::new(),
        }
    }

    /// Adds `text`. Returns what can be emitted now, and whether a stop string was completed, in
    /// which case nothing after the match is ever emitted. When two stop strings complete in the
    /// same text, the earlier one wins.
    pub(crate) fn push<'t>(&mut self, text: &'t str) -> (Cow<'t, str>, bool) {
        if self.stops.is_empty() {
            return (Cow::Borrowed(text), false);
        }
        self.held.push_str(text);
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min();
        if let Some(pos) = hit {
            self.held.truncate(pos);
            return (Cow::Owned(std::mem::take(&mut self.held)), true);
        }
        let keep = self.partial_match_start();
        let rest = self.held.split_off(keep);
        (Cow::Owned(std::mem::replace(&mut self.held, rest)), false)
    }

    /// Text still held when generation ended without completing a match.
    pub(crate) fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Where the longest tail of `held` that begins some stop string starts.
    fn partial_match_start(&self) -> usize {
        self.held
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| {
                let tail = &self.held[i..];
                self.stops.iter().any(|s| s.starts_with(tail))
            })
            .unwrap_or(self.held.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pushes `chunks`, returning the emitted text after each and whether a match was completed.
    fn run(stops: &[&str], chunks: &[&str]) -> (Vec<String>, bool) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut m = StopMatcher::new(&stops);
        let mut out = Vec::new();
        for c in chunks {
            let (ready, hit) = m.push(c);
            out.push(ready.into_owned());
            if hit {
                return (out, true);
            }
        }
        out.push(m.finish());
        (out, false)
    }

    #[test]
    fn without_stops_text_passes_through_unchanged_and_unbuffered() {
        let (out, hit) = run(&[], &["ab", "cd"]);
        assert_eq!(
            (out, hit),
            (vec!["ab".into(), "cd".into(), "".into()], false)
        );
    }

    #[test]
    fn match_inside_one_chunk_cuts_the_text() {
        let (out, hit) = run(&["STOP"], &["abSTOPcd"]);
        assert_eq!((out, hit), (vec!["ab".to_owned()], true));
    }

    #[test]
    fn match_split_across_chunks_is_held_back_then_dropped() {
        let (out, hit) = run(&["STOP"], &["abST", "O", "Pcd"]);
        assert_eq!(
            (out, hit),
            (vec!["ab".to_owned(), "".into(), "".into()], true)
        );
    }

    #[test]
    fn partial_match_that_fails_is_released() {
        let (out, hit) = run(&["STOP"], &["abST", "x"]);
        assert_eq!(out, ["ab", "STx", ""]);
        assert!(!hit);
    }

    #[test]
    fn partial_match_is_released_when_generation_ends() {
        let (out, hit) = run(&["STOP"], &["abST"]);
        assert_eq!(out, ["ab", "ST"]);
        assert!(!hit);
    }

    #[test]
    fn text_that_cannot_start_a_stop_is_not_delayed() {
        let (out, _) = run(&["STOP"], &["hello ", "world"]);
        assert_eq!(out, ["hello ", "world", ""]);
    }

    #[test]
    fn failed_partial_match_can_begin_another() {
        // "aab": the first "a" fails, but the second can still begin the match.
        let (out, hit) = run(&["ab"], &["a", "a", "b"]);
        assert_eq!(out, ["", "a", ""]);
        assert!(hit);
    }

    #[test]
    fn earliest_of_several_matches_wins() {
        let (out, hit) = run(&["cd", "ab"], &["xabycdz"]);
        assert_eq!((out, hit), (vec!["x".to_owned()], true));
    }

    #[test]
    fn overlapping_stops_end_at_the_first_one_completed() {
        // "bc" completes while "abcd" still needs a "d": the reply can't wait on it.
        let (out, hit) = run(&["abcd", "bc"], &["ab", "c"]);
        assert_eq!(out, ["", "a"]);
        assert!(hit);
    }

    #[test]
    fn match_at_the_very_start_emits_nothing() {
        let (out, hit) = run(&["ab"], &["abc"]);
        assert_eq!((out, hit), (vec![String::new()], true));
    }

    #[test]
    fn multi_byte_text_splits_only_on_character_boundaries() {
        let (out, hit) = run(&["日本語"], &["é日", "本", "語ü"]);
        assert_eq!(out, ["é", "", ""]);
        assert!(hit);
        let (out, hit) = run(&["日本語"], &["é日本", "x"]);
        assert_eq!(out, ["é", "日本x", ""]);
        assert!(!hit);
    }
}
