GESTURES = {
    "palm": {
        "scale": True,
    },
    "swipe_down": {
        "scale": True,
    },
    "swipe_up": {
        "src": "swipe_down",
        "scale": True,
    },
    "swipe_right": {
        "scale": True,
    },
    "swipe_left": {
        "src": "swipe_right",
        "scale": True,
    },
    "pinch": {
        "scale": True,
    },
    "expand": {
        "src": "pinch",
        "scale": True,
    },
    "flick_left": {
        "scale": True,
    },
    "flick_right": {
        "src": "flick_left",
        "scale": True,
    },
    "fist": {
        "scale": True,
    },
}

for id,gesture in enumerate(GESTURES):
  GESTURES[gesture]["id"] = id