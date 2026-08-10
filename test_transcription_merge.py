import unittest

from meeting_helper_modern_gui import merge_transcription_chunks


class MergeTranscriptionChunksTest(unittest.TestCase):
    def test_sequential_fragments(self):
        self.assertEqual(
            merge_transcription_chunks("Привет коллеги", "обсудим план"),
            "Привет коллеги обсудим план",
        )

    def test_overlapping_fragments(self):
        self.assertEqual(
            merge_transcription_chunks(
                "Сегодня обсудим запуск проекта",
                "запуск проекта и сроки поставки",
            ),
            "Сегодня обсудим запуск проекта и сроки поставки",
        )

    def test_repeated_last_word(self):
        self.assertEqual(
            merge_transcription_chunks("Нужно проверить бюджет", "бюджет на квартал"),
            "Нужно проверить бюджет на квартал",
        )

    def test_single_word_boundary_repeat_with_punctuation(self):
        self.assertEqual(
            merge_transcription_chunks("Ясно, нужно проверить", "проверить задачу"),
            "Ясно, нужно проверить задачу",
        )

    def test_multiple_word_boundary_repeat(self):
        self.assertEqual(
            merge_transcription_chunks(
                "На самом деле нужно проверить доступы",
                "нужно проверить доступы и роли",
            ),
            "На самом деле нужно проверить доступы и роли",
        )

    def test_fragment_inside_existing_text(self):
        self.assertEqual(
            merge_transcription_chunks(
                "Коллеги уже обсудили бюджет и сроки",
                "обсудили бюджет",
            ),
            "Коллеги уже обсудили бюджет и сроки",
        )

    def test_normal_repetition_inside_new_fragment_is_preserved(self):
        self.assertEqual(
            merge_transcription_chunks("Он сказал", "да да надо делать"),
            "Он сказал да да надо делать",
        )

    def test_raw_ellipsis_suffix_is_removed(self):
        self.assertEqual(
            merge_transcription_chunks("Если там есть проблема", "проблема там..."),
            "Если там есть проблема там",
        )

    def test_empty_fragment(self):
        self.assertEqual(
            merge_transcription_chunks("Уже есть текст", "..."),
            "Уже есть текст",
        )

    def test_long_text(self):
        first = " ".join(f"слово{i}" for i in range(80))
        second = " ".join(f"слово{i}" for i in range(75, 100))
        expected = " ".join(f"слово{i}" for i in range(100))
        self.assertEqual(merge_transcription_chunks(first, second), expected)


if __name__ == "__main__":
    unittest.main()
