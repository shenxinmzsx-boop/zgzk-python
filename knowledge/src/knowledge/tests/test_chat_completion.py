import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from openai import OpenAI

from knowledge.utils.chat_completion import generate_answer


class GenerateAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock(spec=OpenAI)
        self.messages = [
            {"role": "system", "content": "只根据资料回答"},
            {"role": "user", "content": "问题"},
        ]

    def test_returns_trimmed_answer(self) -> None:
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="  根据资料回答。  ")
                )
            ]
        )

        answer = generate_answer(
            self.client,
            model="chat-model",
            messages=self.messages,
        )

        self.assertEqual(answer, "根据资料回答。")
        self.client.chat.completions.create.assert_called_once_with(
            model="chat-model",
            messages=self.messages,
        )

    def test_rejects_invalid_inputs_without_calling_api(self) -> None:
        invalid_arguments = [
            {"model": " ", "messages": self.messages},
            {"model": "chat-model", "messages": []},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    generate_answer(self.client, **arguments)

        self.client.chat.completions.create.assert_not_called()

    def test_rejects_missing_answer_choice(self) -> None:
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[]
        )

        with self.assertRaisesRegex(RuntimeError, "未返回回答选项"):
            generate_answer(
                self.client,
                model="chat-model",
                messages=self.messages,
            )

    def test_rejects_multiple_answer_choices(self) -> None:
        choice = SimpleNamespace(message=SimpleNamespace(content="回答"))
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[choice, choice]
        )

        with self.assertRaisesRegex(RuntimeError, "实际返回2"):
            generate_answer(
                self.client,
                model="chat-model",
                messages=self.messages,
            )

    def test_rejects_empty_answer_content(self) -> None:
        for content in (None, " "):
            with self.subTest(content=content):
                self.client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=content)
                        )
                    ]
                )

                with self.assertRaisesRegex(RuntimeError, "回答为空"):
                    generate_answer(
                        self.client,
                        model="chat-model",
                        messages=self.messages,
                    )
