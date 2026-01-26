"""Tests for MultiAgent implementation."""
import sys
sys.path.insert(0, '.')

from consciousness.ml_research.ml_techniques.agentic import (
    MultiAgent, AgentSpec, CoordinationPattern, MessageType, ConsensusProtocol
)


def test_debate_pattern():
    """Test debate coordination pattern."""
    print('=== Testing MultiAgent - Debate Pattern ===')
    ma = MultiAgent(coordination='debate', max_rounds=2)
    result = ma.run('Should we use Python or Rust for a new CLI tool?')

    assert result.success, f"Debate failed: {result.error}"
    assert result.metadata.get("coordination_pattern") == "debate"
    assert result.metadata.get("message_count", 0) > 0
    assert "final_proposal" in result.output
    assert "judgment" in result.output

    print(f'  Success: {result.success}')
    print(f'  Pattern: {result.metadata.get("coordination_pattern")}')
    print(f'  Messages: {result.metadata.get("message_count")}')
    print('  PASSED')


def test_division_pattern():
    """Test division coordination pattern."""
    print('=== Testing MultiAgent - Division Pattern ===')
    ma = MultiAgent(coordination='division', max_rounds=1)
    result = ma.run('Build a web scraper that extracts product prices')

    assert result.success, f"Division failed: {result.error}"
    assert result.metadata.get("coordination_pattern") == "division"
    assert "worker_results" in result.output
    assert "aggregated_result" in result.output

    print(f'  Success: {result.success}')
    print(f'  Pattern: {result.metadata.get("coordination_pattern")}')
    print(f'  Workers used: {result.output.get("workers_used")}')
    print('  PASSED')


def test_hierarchical_pattern():
    """Test hierarchical coordination pattern."""
    print('=== Testing MultiAgent - Hierarchical Pattern ===')
    ma = MultiAgent(coordination='hierarchical', max_rounds=2)
    result = ma.run('Create a marketing plan for a new product')

    assert result.success, f"Hierarchical failed: {result.error}"
    assert result.metadata.get("coordination_pattern") == "hierarchical"
    assert "final_summary" in result.output

    print(f'  Success: {result.success}')
    print(f'  Pattern: {result.metadata.get("coordination_pattern")}')
    print(f'  Rounds: {result.output.get("rounds_completed")}')
    print('  PASSED')


def test_peer_review_pattern():
    """Test peer review coordination pattern."""
    print('=== Testing MultiAgent - Peer Review Pattern ===')
    ma = MultiAgent(coordination='peer_review', max_rounds=2)
    result = ma.run('Write a haiku about programming')

    assert result.success, f"Peer review failed: {result.error}"
    assert result.metadata.get("coordination_pattern") == "peer_review"
    assert "final_work" in result.output
    assert "review_rounds" in result.output

    print(f'  Success: {result.success}')
    print(f'  Review rounds: {result.output.get("review_rounds")}')
    print('  PASSED')


def test_consensus_protocol():
    """Test ConsensusProtocol utilities."""
    print('=== Testing ConsensusProtocol ===')

    # Majority vote
    responses = ['A', 'B', 'A', 'A', 'C']
    winner, conf = ConsensusProtocol.majority_vote(responses)
    assert winner == 'A'
    assert conf == 0.6
    print(f'  Majority vote: {winner} (confidence: {conf})')

    # Unanimous
    unanimous, val = ConsensusProtocol.unanimous(['same', 'same', 'same'])
    assert unanimous is True
    assert val == 'same'
    print(f'  Unanimous: {unanimous}, value: {val}')

    # Not unanimous
    not_unan, _ = ConsensusProtocol.unanimous(['a', 'b', 'a'])
    assert not_unan is False
    print(f'  Not unanimous: {not_unan}')

    print('  PASSED')


def test_custom_agents():
    """Test with custom agent specifications."""
    print('=== Testing Custom Agents ===')
    agents = [
        AgentSpec("expert1", "proposer", "Domain expert", system_prompt="You are a domain expert."),
        AgentSpec("skeptic", "critic", "Skeptical reviewer", system_prompt="You are skeptical."),
        AgentSpec("arbiter", "judge", "Fair arbiter", system_prompt="You judge fairly."),
    ]
    ma = MultiAgent(agents=agents, coordination='debate', max_rounds=1)
    result = ma.run('Evaluate this architecture decision')

    assert result.success
    assert "expert1" in result.metadata.get("agents_used", [])
    print(f'  Agents used: {result.metadata.get("agents_used")}')
    print('  PASSED')


if __name__ == '__main__':
    print('=' * 60)
    print('MultiAgent Test Suite')
    print('=' * 60)

    test_debate_pattern()
    print()
    test_division_pattern()
    print()
    test_hierarchical_pattern()
    print()
    test_peer_review_pattern()
    print()
    test_consensus_protocol()
    print()
    test_custom_agents()

    print()
    print('=' * 60)
    print('All MultiAgent tests PASSED!')
    print('=' * 60)
