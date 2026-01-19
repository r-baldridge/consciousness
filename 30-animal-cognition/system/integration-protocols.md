# Animal Cognition Integration Protocols

## Overview
Communication protocols for integrating animal cognition with other consciousness forms.

## Inter-Module Communication

### Message Types
```python
class AnimalCognitionMessageTypes:
    COGNITION_QUERY = "animal_cognition_query"
    COGNITION_RESPONSE = "animal_cognition_response"
    SPECIES_COMPARISON = "species_comparison_request"
    CONSCIOUSNESS_INDICATOR = "consciousness_indicator_query"
    INDIGENOUS_ANIMAL_LINK = "indigenous_animal_knowledge_link"
```

### Primary Integration Points

#### Form 28 (Philosophy) Integration
- Philosophy of mind implications for animal consciousness
- Moral status of animals connects to ethics domain
- Consciousness studies cross-reference

#### Form 29 (Folk Wisdom) Integration
- IndigenousAnimalKnowledge links to FolkWisdom teachings
- Traditional ecological knowledge validation
- Behavioral observations corroboration

### Communication Protocol
```python
class AnimalCognitionCommunicationProtocol:
    def send_cognition_query(self, species, domains):
        return {
            'species_id': species.id,
            'taxonomic_group': species.group,
            'cognition_domains': domains,
            'evidence_types': ['behavioral', 'neural', 'indigenous']
        }
```

## Performance Metrics
- Message latency: < 50ms
- Cross-species comparison coherence: > 0.85
