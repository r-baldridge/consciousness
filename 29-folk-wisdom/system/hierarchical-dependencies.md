# Folk Wisdom Hierarchical Dependencies

## Overview
Dependencies and outputs for Form 29 (Folk Wisdom) within the consciousness hierarchy.

## Dependencies

### Required Forms
| Form | Dependency Type | Purpose |
|------|-----------------|---------|
| 08-arousal | Gating | Consciousness level determines wisdom accessibility |
| 13-integrated-information | Integration | Phi-weighted wisdom synthesis |
| 14-global-workspace | Broadcasting | Global access to folk insights |
| 28-philosophy | Conceptual | Philosophical framework for wisdom interpretation |

### Optional Dependencies
| Form | Dependency Type | Purpose |
|------|-----------------|---------|
| 07-emotional | Enhancement | Emotional resonance with wisdom |
| 12-narrative | Enhancement | Storytelling tradition processing |

## Outputs Provided

### To Other Forms
```python
class FolkWisdomOutputs:
    outputs = {
        'to_form_28': ['folk_philosophical_bridges', 'oral_epistemology'],
        'to_form_30': ['indigenous_animal_wisdom', 'traditional_ecological_knowledge'],
        'to_form_36': ['contemplative_folk_practices', 'indigenous_meditation'],
        'to_global_workspace': ['cultural_insights', 'traditional_guidance']
    }
```

## Dependency Resolution
- Graceful degradation if optional dependencies unavailable
- Core function maintained with required dependencies only
