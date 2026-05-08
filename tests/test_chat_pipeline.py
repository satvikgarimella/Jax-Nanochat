import asyncio
import unittest
from unittest import mock

import chat_pipeline


class ChatPipelineTests(unittest.TestCase):
    def test_try_solve_math_handles_basic_expression(self):
        self.assertEqual(chat_pipeline.try_solve_math("what is 2 + 2?"), "4")

    def test_build_extractive_answer_prefers_relevant_sentences(self):
        context = (
            "Machine learning is a branch of artificial intelligence focused on learning "
            "patterns from data to make predictions or decisions. "
            "It is widely used in areas like recommendation systems and fraud detection. "
            "Many systems improve as they see more examples."
        )

        answer = chat_pipeline.build_extractive_answer("What is machine learning?", context)

        self.assertIsNotNone(answer)
        self.assertIn("Machine learning", answer)
        self.assertIn("data", answer)

    def test_build_extractive_answer_prefers_definition_over_supporting_snippet(self):
        context = (
            "This pattern recognition ability enables machine learning models to make decisions or "
            "predictions without explicit, hard-coded instructions. [1] "
            "Advances in the field of deep learning have allowed neural networks, a class of "
            "statistical algorithms, to surpass many previous machine learning approaches in performance. "
            "Machine learning is a field of study in artificial intelligence concerned with the development "
            "and study of statistical algorithms that can learn from data and generalize to unseen data."
        )

        answer = chat_pipeline.build_extractive_answer("What is machine learning?", context)

        self.assertIsNotNone(answer)
        self.assertTrue(answer.lower().startswith("machine learning is"))
        self.assertNotIn("[1]", answer)

    def test_prepare_model_payload_uses_guided_definition_for_common_terms(self):
        direct_response, body, mode = asyncio.run(
            chat_pipeline.prepare_model_payload(
                {"message": "What is machine learning?", "history": [], "temperature": 0.9, "max_tokens": 300}
            )
        )

        self.assertIsNone(direct_response)
        self.assertEqual(mode, "guided_definition")
        self.assertIn("Reference definition:", body["message"])
        self.assertEqual(
            body["_canonical_answer"],
            "Machine learning is a branch of AI where algorithms learn patterns from data so they can make predictions or decisions without being explicitly programmed for every case.",
        )
        self.assertEqual(body["temperature"], 0.55)
        self.assertEqual(body["max_tokens"], 120)

    def test_prepare_model_payload_uses_extractive_answer_for_unknown_definitions(self):
        async def run_test():
            with mock.patch.object(
                chat_pipeline,
                "fetch_web_context",
                new=mock.AsyncMock(
                    return_value=(
                        "Gradient descent is an optimization method that updates parameters in the "
                        "direction that reduces error. It is widely used to train machine learning models."
                    )
                ),
            ):
                return await chat_pipeline.prepare_model_payload(
                    {"message": "What is gradient descent?", "history": [], "temperature": 0.9, "max_tokens": 300}
                )

        direct_response, body, mode = asyncio.run(run_test())

        self.assertEqual(mode, "extractive")
        self.assertIsNotNone(direct_response)
        self.assertIn("Gradient descent", direct_response)
        self.assertEqual(body["temperature"], 0.55)
        self.assertEqual(body["max_tokens"], 120)

    def test_build_extractive_answer_skips_question_echo_support(self):
        context = (
            "The capital of Canada is Ottawa in the province of Ontario. "
            "What is the capital of Canada, and where is it located? "
            "Ottawa is located in southeastern Ontario."
        )

        answer = chat_pipeline.build_extractive_answer(
            "What is the capital of Canada, and where is it located?",
            context,
        )

        self.assertIsNotNone(answer)
        self.assertNotIn("What is the capital of Canada", answer)
        self.assertIn("Ottawa", answer)

    def test_should_use_extractive_answer_is_narrow(self):
        self.assertTrue(chat_pipeline.should_use_extractive_answer("What is machine learning?"))
        self.assertFalse(
            chat_pipeline.should_use_extractive_answer(
                "What is the capital of Canada, and where is it located?"
            )
        )

    def test_get_canonical_definition_handles_aliases(self):
        self.assertEqual(
            chat_pipeline.get_canonical_definition("What is tensor flow?"),
            "TensorFlow is an open-source machine learning framework from Google for building and training neural networks.",
        )

    def test_finalize_model_response_uses_canonical_as_safety_net(self):
        final = chat_pipeline.finalize_model_response(
            "What is machine learning?",
            (
                "The topic of machine learning has been a subject of ongoing debate and research, "
                "and there is no clear consensus on the topic or value of machine learning."
            ),
            chat_pipeline.get_canonical_definition("What is machine learning?"),
        )

        self.assertEqual(
            final,
            "Machine learning is a branch of AI where algorithms learn patterns from data so they can make predictions or decisions without being explicitly programmed for every case.",
        )

    def test_finalize_model_response_keeps_reasonable_definition_answer(self):
        final = chat_pipeline.finalize_model_response(
            "What is machine learning?",
            "Machine learning is a branch of AI that learns from data to make predictions or decisions.",
            chat_pipeline.get_canonical_definition("What is machine learning?"),
        )

        self.assertEqual(
            final,
            "Machine learning is a branch of AI that learns from data to make predictions or decisions.",
        )


if __name__ == "__main__":
    unittest.main()
