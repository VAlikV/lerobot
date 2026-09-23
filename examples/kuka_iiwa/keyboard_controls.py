# Overriding the init_keyboard_listener class from 
# src/lerobot/utils/keyboard_input.py to add more control keys

from lerobot.utils.keyboard_input import create_key_listener, apply_recording_control

def init_kuka_keyboard_listener():
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,

        "apply_scale": False,
    }

    def on_key(name: str):
        key = name.lower()

        if key in ("right", "n"):
            apply_recording_control("right", events)

        elif key in ("left", "r"):
            apply_recording_control("left", events)

        elif key in ("esc", "q"):
            apply_recording_control("esc", events)

        elif key == "s":
            events["apply_scale"] = True

    listener = create_key_listener(
        on_key,
        controls_help="Right/Left/Esc, n=next, r=re-record, q=quit, s=apply_scale",
    )

    return listener, events