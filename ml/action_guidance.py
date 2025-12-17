"""Action guidance for each plastic type."""

ACTION_GUIDANCE = {
    "HDPE": [
        "Rinse thoroughly with water",
        "Remove any labels or caps if possible",
        "Check local recycling guidelines"
    ],
    "PET": [
        "Rinse and remove cap",
        "Flatten to save space",
        "Remove label if possible"
    ],
    "PP": [
        "Rinse if used for food",
        "Check label for specific instructions",
        "Avoid if oily or contaminated"
    ],
    "PS": [
        "Do not burn or incinerate",
        "Check if local facility accepts",
        "Consider alternative disposal if not recyclable"
    ],
    "OTHER": [
        "Check local recycling guidelines",
        "Contact waste management for guidance",
        "Consider reuse if possible"
    ]
}

def get_action_guidance(plastic_type: str) -> list:
    """
    Get action guidance for a plastic type.
    
    Args:
        plastic_type: Plastic type label
        
    Returns:
        List of action guidance strings
    """
    return ACTION_GUIDANCE.get(plastic_type, ACTION_GUIDANCE["OTHER"])

