import pytest
from pydantic import ValidationError

from computer_use.schema import Action, ActionBatch


def test_action_batch_happy_path():
    batch = ActionBatch(actions=[Action(kind="click", selector="#go")])
    assert len(batch.actions) == 1
    assert batch.actions[0].kind == "click"


def test_action_invalid_kind_edge_case():
    with pytest.raises(ValidationError):
        Action(kind="explode")
