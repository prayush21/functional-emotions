import torch

from functional_emotions.prototype1 import (
    _generation_malformed_reason,
    _generation_prompt,
    binary_auc,
    difference_in_means,
    evaluate_vectors,
    flat_nonzero,
    generation_failure_report,
    generation_inputs,
    neutral_pca,
    pooling_diagnostics,
    remove_components,
    resolve_evaluation_layer,
    split_topics,
    validate_story_rows,
    write_generation_failure_diagnostics,
)


def test_topic_split_is_deterministic_and_disjoint():
    topics = [f"topic-{index}" for index in range(8)]
    train_a, test_a = split_topics(topics, 0.75, 42)
    train_b, test_b = split_topics(topics, 0.75, 42)

    assert (train_a, test_a) == (train_b, test_b)
    assert train_a.isdisjoint(test_a)
    assert train_a | test_a == set(topics)


def test_story_audit_detects_whole_word_label_leakage():
    rows = [
        {"topic": "one", "emotion": "happy", "text": "She was happy with the result."},
        {"topic": "two", "emotion": "sad", "text": "He stared at the empty chair."},
    ]
    audit = validate_story_rows(rows, ["happy", "sad"], {"sad": ["sorrow"]})

    assert audit["lexical_leakage"] == [{"row": 0, "emotion": "happy", "terms": ["happy"]}]


def test_story_audit_detects_simple_morphological_leakage():
    rows = [{"topic": "one", "emotion": "happy", "text": "She smiled happily at the result."}]
    audit = validate_story_rows(rows, ["happy"], {})

    assert audit["lexical_leakage"] == [{"row": 0, "emotion": "happy", "terms": ["happy"]}]


def test_pooling_diagnostics_report_final_token_fallbacks():
    class TinyTokenizer:
        def __call__(self, text, truncation=True):
            return {"input_ids": text.split()}

    rows = [
        {"id": "a", "topic": "one", "emotion": "happy", "text": "one two"},
        {"id": "b", "topic": "two", "emotion": "sad", "text": "one two three four"},
    ]
    diagnostics = pooling_diagnostics(TinyTokenizer(), rows, token_start=3, split_name="test")

    assert diagnostics["final_token_fallback_rows"] == 1
    assert diagnostics["fallback_examples"][0]["id"] == "a"
    assert diagnostics["per_emotion"]["happy"]["fallback_rows"] == 1


def test_emotion_generation_prompt_uses_behavioral_guidance():
    prompt = _generation_prompt("A student receives an email", "happy", ["joy"], False)

    assert "experiencing happy" not in prompt
    assert "plausible context" in prompt
    assert "Forbidden words" in prompt


def test_unterminated_think_output_is_malformed():
    assert _generation_malformed_reason("<think>reasoning without an end") == "unterminated_think"
    assert _generation_malformed_reason("<think>done</think>story") is None


def test_generation_inputs_accept_chat_template_tensor():
    class ChatTokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
            return torch.tensor([[1, 2, 3]])

    input_ids, attention_mask = generation_inputs(ChatTokenizer(), "prompt", "cpu")

    assert torch.equal(input_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(attention_mask, torch.tensor([[1, 1, 1]]))


def test_generation_inputs_passes_enable_thinking_when_supported():
    class ChatTokenizer:
        chat_template = "template"

        def __init__(self):
            self.enable_thinking = None

        def apply_chat_template(
            self, messages, add_generation_prompt, return_tensors, enable_thinking=True
        ):
            self.enable_thinking = enable_thinking
            return torch.tensor([[1, 2, 3]])

    tokenizer = ChatTokenizer()
    generation_inputs(tokenizer, "prompt", "cpu", enable_thinking=False)

    assert tokenizer.enable_thinking is False


def test_generation_inputs_accept_chat_template_batch_encoding_shape():
    class ChatBatch(dict):
        def to(self, device):
            return self

    class ChatTokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
            return ChatBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    input_ids, attention_mask = generation_inputs(ChatTokenizer(), "prompt", "cpu")

    assert torch.equal(input_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(attention_mask, torch.tensor([[1, 1, 1]]))


def test_generation_failure_report_counts_matched_terms():
    report = generation_failure_report(
        topic="A student finds a map",
        emotion="happy",
        story_index=0,
        sequence=0,
        maximum_attempts=2,
        forbidden_terms=["joy"],
        prompt="prompt",
        attempts=[
            {"attempt": 1, "matched_terms": ["happy"], "text": "happy text"},
            {"attempt": 2, "matched_terms": ["joy"], "text": "joy text"},
        ],
        successful_rows=0,
    )

    assert report["forbidden_terms"] == ["happy", "joy"]
    assert report["matched_term_counts"] == {"happy": 1, "joy": 1}


def test_generation_failure_diagnostics_are_written(tmp_path):
    dataset_path = tmp_path / "stories.jsonl"
    report = {"topic": "topic", "attempts": []}
    partial_rows = [{"id": "story-000000", "topic": "topic", "emotion": "happy", "text": "ok"}]

    failure_path = write_generation_failure_diagnostics(dataset_path, report, partial_rows)

    assert failure_path == tmp_path / "generation_failure.json"
    assert failure_path.is_file()
    assert (tmp_path / "generation_partial_stories.jsonl").read_text().strip()


def test_difference_in_means_uses_across_emotion_baseline():
    activations = torch.tensor([[[2.0, 0.0]], [[4.0, 0.0]], [[0.0, 2.0]], [[0.0, 4.0]]])
    vectors = difference_in_means(activations, ["a", "a", "b", "b"], ["a", "b"])

    assert torch.equal(vectors[0], torch.tensor([[1.5, -1.5]]))
    assert torch.equal(vectors[1], -vectors[0])


def test_flat_nonzero_uses_older_torch_compatible_api():
    values = torch.tensor([False, True, False, True])

    assert torch.equal(flat_nonzero(values), torch.tensor([1, 3]))


def test_neutral_pca_removal_is_orthogonal_and_normalized():
    neutral = torch.tensor(
        [
            [[-2.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
    )
    components, metadata = neutral_pca(neutral, 0.5)
    vectors = torch.tensor([[[1.0, 1.0, 0.0]], [[1.0, 0.0, 1.0]]])
    cleaned = remove_components(vectors, components)

    assert metadata[0]["number_of_components"] == 1
    assert torch.allclose(cleaned[..., 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(cleaned.norm(dim=-1), torch.ones(2))


def test_held_out_validation_scores_separable_vectors():
    activations = torch.tensor(
        [
            [[2.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 2.0]],
            [[0.0, 1.0]],
        ]
    )
    vectors = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    result = evaluate_vectors(activations, ["a", "a", "b", "b"], ["a", "b"], vectors)[0]

    assert result["accuracy"] == 1.0
    assert result["macro_auc"] == 1.0
    assert result["mean_correct_margin"] > 0
    assert result["confusion_matrix"] == {
        "labels": ["a", "b"],
        "rows_are_true_labels": [[2, 0], [0, 2]],
    }
    assert binary_auc(torch.tensor([0.0, 1.0]), torch.tensor([False, True])) == 1.0


def test_two_thirds_layer_is_pre_registered():
    assert resolve_evaluation_layer("two_thirds", 28) == 19
