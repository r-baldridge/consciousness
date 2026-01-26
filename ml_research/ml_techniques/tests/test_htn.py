"""Tests for Hierarchical Task Network (HTN) Decomposition."""
import sys
sys.path.insert(0, '.')

from consciousness.ml_research.ml_techniques.decomposition import (
    HierarchicalTaskDecomposition,
    TaskTemplate,
    TaskInstance,
    Method,
    Condition,
    TaskStatus,
    MethodSelectionStrategy,
    HTNPlan,
    PlanValidator,
)


def test_condition_evaluation():
    """Test condition evaluation against state."""
    print('=== Testing Condition Evaluation ===')

    state = {
        "at": {"location": "home"},
        "has_item": {"key", "phone"},
        "door_open": False,
    }

    # Positive condition
    cond1 = Condition("has_item", {"item": "key"})
    assert cond1.evaluate(state), "Should find 'key' in has_item set"
    print(f'  {cond1}: {cond1.evaluate(state)} (expected True)')

    # Negative condition
    cond2 = Condition("has_item", {"item": "wallet"})
    assert not cond2.evaluate(state), "Should not find 'wallet'"
    print(f'  {cond2}: {cond2.evaluate(state)} (expected False)')

    # Negated condition
    cond3 = Condition("door_open", {}, negated=True)
    assert cond3.evaluate(state), "NOT door_open should be True"
    print(f'  {cond3}: {cond3.evaluate(state)} (expected True)')

    print('  PASSED')


def test_task_template():
    """Test TaskTemplate creation and instantiation."""
    print('=== Testing TaskTemplate ===')

    template = TaskTemplate(
        name="pickup",
        parameters=["item"],
        preconditions=[Condition("at_item", {"item": "item"})],
        effects=[Condition("holding", {"item": "item"})],
        is_primitive=True,
        cost=1.0,
    )

    # Instantiate with bindings
    instance = template.instantiate({"item": "book"})

    assert instance.name == "pickup"
    assert instance.bindings == {"item": "book"}
    assert instance.is_primitive
    assert instance.status == TaskStatus.PENDING
    print(f'  Instance: {instance}')
    print('  PASSED')


def test_task_preconditions():
    """Test precondition checking."""
    print('=== Testing Task Preconditions ===')

    template = TaskTemplate(
        name="pickup",
        parameters=["item"],
        preconditions=[
            Condition("at_item", {"item": "item"}),
            Condition("hands_free", {}),
        ],
        is_primitive=True,
    )

    # State where preconditions are met
    good_state = {
        "at_item": {"book"},
        "hands_free": True,
    }
    assert template.check_preconditions(good_state, {"item": "book"})
    print(f'  Good state: preconditions met = True')

    # State where preconditions are not met
    bad_state = {
        "at_item": {"pen"},  # Wrong item
        "hands_free": True,
    }
    assert not template.check_preconditions(bad_state, {"item": "book"})
    print(f'  Bad state: preconditions met = False')

    print('  PASSED')


def test_task_effects():
    """Test applying task effects to state."""
    print('=== Testing Task Effects ===')

    template = TaskTemplate(
        name="pickup",
        parameters=["item"],
        effects=[
            Condition("holding", {"item": "item"}),
            Condition("at_item", {"item": "item"}, negated=True),
        ],
        is_primitive=True,
    )

    initial_state = {
        "at_item": {"book", "pen"},
        "holding": set(),
    }

    new_state = template.apply_effects(initial_state, {"item": "book"})

    assert "book" in new_state["holding"], "Should be holding book"
    assert "book" not in new_state["at_item"], "Book should not be at_item"
    assert "pen" in new_state["at_item"], "Pen should still be at_item"
    print(f'  After pickup(book): holding={new_state["holding"]}, at_item={new_state["at_item"]}')
    print('  PASSED')


def test_method_decomposition():
    """Test method decomposition into subtasks."""
    print('=== Testing Method Decomposition ===')

    # Define task library
    task_library = {
        "goto": TaskTemplate(name="goto", parameters=["location"], is_primitive=True),
        "pickup": TaskTemplate(name="pickup", parameters=["item"], is_primitive=True),
        "putdown": TaskTemplate(name="putdown", parameters=["item"], is_primitive=True),
    }

    # Define delivery method
    deliver_method = Method(
        name="deliver_direct",
        task_name="deliver",
        parameters=["item", "destination"],
        subtask_templates=[
            ("goto", {"location": "item_location"}),
            ("pickup", {"item": "item"}),
            ("goto", {"location": "destination"}),
            ("putdown", {"item": "item"}),
        ],
    )

    # Get subtasks
    bindings = {"item": "package", "destination": "office", "item_location": "warehouse"}
    subtasks = deliver_method.get_subtasks(bindings, task_library)

    assert len(subtasks) == 4
    assert subtasks[0].name == "goto"
    assert subtasks[1].name == "pickup"
    print(f'  Subtasks: {[str(t) for t in subtasks]}')
    print('  PASSED')


def test_htn_simple_decomposition():
    """Test HTN decomposition with a simple task."""
    print('=== Testing HTN Simple Decomposition ===')

    # Define domain
    task_library = {
        "greet": TaskTemplate(
            name="greet",
            parameters=["person"],
            is_primitive=True,
            cost=1.0,
        ),
        "introduce": TaskTemplate(
            name="introduce",
            parameters=["topic"],
            is_primitive=True,
            cost=1.0,
        ),
        "conclude": TaskTemplate(
            name="conclude",
            is_primitive=True,
            cost=1.0,
        ),
    }

    method_library = {
        "present_standard": Method(
            name="present_standard",
            task_name="present",
            parameters=["audience", "topic"],
            subtask_templates=[
                ("greet", {"person": "audience"}),
                ("introduce", {"topic": "topic"}),
                ("conclude", {}),
            ],
            priority=1,
        ),
    }

    htn = HierarchicalTaskDecomposition(
        task_library=task_library,
        method_library=method_library,
        initial_state={},
    )

    result = htn.run({
        "task": "present",
        "bindings": {"audience": "students", "topic": "HTN planning"},
    })

    assert result.success, f"Decomposition failed: {result.error}"
    assert result.output["num_steps"] == 3
    print(f'  Plan: {result.output["plan"]}')
    print(f'  Steps: {result.output["num_steps"]}')
    print('  PASSED')


def test_htn_method_selection():
    """Test method selection with multiple options."""
    print('=== Testing HTN Method Selection ===')

    task_library = {
        "walk": TaskTemplate(name="walk", parameters=["to"], is_primitive=True, cost=10.0),
        "drive": TaskTemplate(name="drive", parameters=["to"], is_primitive=True, cost=2.0),
        "get_in_car": TaskTemplate(name="get_in_car", is_primitive=True, cost=0.5),
        "get_out_car": TaskTemplate(name="get_out_car", is_primitive=True, cost=0.5),
    }

    method_library = {
        "travel_walk": Method(
            name="travel_walk",
            task_name="travel",
            subtask_templates=[("walk", {"to": "destination"})],
            cost=10.0,
            priority=0,
        ),
        "travel_drive": Method(
            name="travel_drive",
            task_name="travel",
            preconditions=[Condition("has_car", {})],
            subtask_templates=[
                ("get_in_car", {}),
                ("drive", {"to": "destination"}),
                ("get_out_car", {}),
            ],
            cost=3.0,
            priority=1,
        ),
    }

    # Without car - should walk
    htn_no_car = HierarchicalTaskDecomposition(
        task_library=task_library,
        method_library=method_library,
        initial_state={},  # No car
        selection_strategy=MethodSelectionStrategy.HIGHEST_PRIORITY,
    )

    result_no_car = htn_no_car.run({
        "task": "travel",
        "bindings": {"destination": "store"},
    })

    assert result_no_car.success
    assert result_no_car.output["num_steps"] == 1  # Just walk
    print(f'  Without car: {result_no_car.output["plan"]}')

    # With car - should drive (higher priority)
    htn_with_car = HierarchicalTaskDecomposition(
        task_library=task_library,
        method_library=method_library,
        initial_state={"has_car": True},
        selection_strategy=MethodSelectionStrategy.HIGHEST_PRIORITY,
    )

    result_with_car = htn_with_car.run({
        "task": "travel",
        "bindings": {"destination": "store"},
    })

    assert result_with_car.success
    assert result_with_car.output["num_steps"] == 3  # get_in, drive, get_out
    print(f'  With car: {result_with_car.output["plan"]}')
    print('  PASSED')


def test_htn_nested_decomposition():
    """Test nested task decomposition."""
    print('=== Testing HTN Nested Decomposition ===')

    task_library = {
        "step1": TaskTemplate(name="step1", is_primitive=True),
        "step2": TaskTemplate(name="step2", is_primitive=True),
        "step3": TaskTemplate(name="step3", is_primitive=True),
        "step4": TaskTemplate(name="step4", is_primitive=True),
        "inner_task": TaskTemplate(name="inner_task", is_primitive=False),  # Compound task
    }

    method_library = {
        "outer_method": Method(
            name="outer_method",
            task_name="outer_task",
            subtask_templates=[
                ("step1", {}),
                ("inner_task", {}),  # Compound subtask
                ("step4", {}),
            ],
        ),
        "inner_method": Method(
            name="inner_method",
            task_name="inner_task",
            subtask_templates=[
                ("step2", {}),
                ("step3", {}),
            ],
        ),
    }

    htn = HierarchicalTaskDecomposition(
        task_library=task_library,
        method_library=method_library,
    )

    result = htn.run("outer_task")

    assert result.success
    assert result.output["num_steps"] == 4  # step1, step2, step3, step4
    print(f'  Plan: {result.output["plan"]}')
    print(f'  Max depth: {result.metadata["max_depth_reached"]}')
    print('  PASSED')


def test_htn_llm_generation():
    """Test LLM-based method generation for unknown tasks."""
    print('=== Testing HTN LLM Method Generation ===')

    # Create HTN without predefined methods but with a backend (placeholder)
    def placeholder_backend(prompt):
        # Return a simple decomposition
        return "SUBTASK: prepare_ingredients()\nSUBTASK: brew_coffee()\nSUBTASK: serve_coffee()"

    htn = HierarchicalTaskDecomposition(
        task_library={},
        method_library={},
        backend=placeholder_backend,
    )

    # This should trigger LLM generation
    result = htn.run("make_coffee")

    assert result.success, f"Should generate decomposition: {result.error}"
    assert result.output["num_steps"] >= 2  # Should have at least 2 steps
    print(f'  Generated plan: {result.output["plan"]}')
    print('  PASSED')


def test_plan_validator():
    """Test plan validation."""
    print('=== Testing Plan Validator ===')

    task_library = {
        "unlock": TaskTemplate(
            name="unlock",
            preconditions=[Condition("has_key", {})],
            effects=[Condition("unlocked", {})],
            is_primitive=True,
        ),
        "open": TaskTemplate(
            name="open",
            preconditions=[Condition("unlocked", {})],
            effects=[Condition("door_open", {})],
            is_primitive=True,
        ),
    }

    # Valid plan: unlock then open
    unlock_task = task_library["unlock"].instantiate({})
    open_task = task_library["open"].instantiate({})

    root = TaskInstance(
        template=TaskTemplate(name="enter", is_primitive=False),
        subtasks=[unlock_task, open_task],
    )

    plan = HTNPlan(
        root_task=root,
        primitive_tasks=[unlock_task, open_task],
    )

    # Validate with key
    initial_state = {"has_key": True}
    is_valid, errors = PlanValidator.validate_plan(plan, initial_state, task_library)

    assert is_valid, f"Plan should be valid: {errors}"
    print(f'  With key: valid = {is_valid}')

    # Validate without key
    initial_state_no_key = {}
    is_valid2, errors2 = PlanValidator.validate_plan(plan, initial_state_no_key, task_library)

    assert not is_valid2, "Plan should be invalid without key"
    print(f'  Without key: valid = {is_valid2}, errors = {errors2[0][:50]}...')
    print('  PASSED')


def test_task_tree_serialization():
    """Test task tree serialization."""
    print('=== Testing Task Tree Serialization ===')

    task_library = {
        "step1": TaskTemplate(name="step1", is_primitive=True),
        "step2": TaskTemplate(name="step2", is_primitive=True),
    }

    method_library = {
        "main_method": Method(
            name="main_method",
            task_name="main",
            subtask_templates=[("step1", {}), ("step2", {})],
        ),
    }

    htn = HierarchicalTaskDecomposition(
        task_library=task_library,
        method_library=method_library,
    )

    result = htn.run("main")

    assert "task_tree" in result.output
    tree = result.output["task_tree"]

    assert tree["name"] == "main"
    assert len(tree["subtasks"]) == 2
    assert tree["subtasks"][0]["is_primitive"]
    print(f'  Tree root: {tree["name"]}')
    print(f'  Subtasks: {[st["name"] for st in tree["subtasks"]]}')
    print('  PASSED')


if __name__ == '__main__':
    print('=' * 60)
    print('HTN Decomposition Test Suite')
    print('=' * 60)

    test_condition_evaluation()
    print()
    test_task_template()
    print()
    test_task_preconditions()
    print()
    test_task_effects()
    print()
    test_method_decomposition()
    print()
    test_htn_simple_decomposition()
    print()
    test_htn_method_selection()
    print()
    test_htn_nested_decomposition()
    print()
    test_htn_llm_generation()
    print()
    test_plan_validator()
    print()
    test_task_tree_serialization()

    print()
    print('=' * 60)
    print('All HTN Decomposition tests PASSED!')
    print('=' * 60)
