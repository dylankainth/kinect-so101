"""Always-on-top STOP / PARK / LIMP panel for the SO-101.

    pythonw -m core.tools.stop_gui

STOP freezes the arm in place (holds). PARK, after a stop, folds it back to rest and goes limp.
LIMP drops torque immediately (the arm falls). If no runner owns the port, LIMP writes
Torque_Enable=0 to the servos directly. Space or Esc also presses STOP.
"""
import tkinter as tk

from core import control

PORT = "COM4"


def direct_limp():
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    bus = FeetechMotorsBus(port=PORT, motors={n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES)
                                              for i, n in enumerate(names)})
    bus.connect(handshake=False)
    try:
        for n in names:
            bus.write("Torque_Enable", n, 0, normalize=False)
    finally:
        bus.disconnect(disable_torque=False)


class Panel:
    def __init__(self, root):
        self.root = root
        root.title("SO-101 STOP")
        root.attributes("-topmost", True)
        root.geometry("320x330+40+40")
        root.configure(bg="#222")
        tk.Button(root, text="STOP", font=("Segoe UI", 40, "bold"), bg="#d11", fg="white",
                  activebackground="#f33", command=self.stop).pack(fill="x", padx=10, pady=(10, 6), ipady=20)
        row = tk.Frame(root, bg="#222")
        row.pack(fill="x", padx=10)
        tk.Button(row, text="PARK", font=("Segoe UI", 14, "bold"), bg="#e90", command=self.park
                  ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        tk.Button(row, text="LIMP", font=("Segoe UI", 14, "bold"), bg="#555", fg="white", command=self.limp
                  ).pack(side="left", expand=True, fill="x", padx=(4, 0))
        tk.Button(root, text="clear (allow new runs)", command=self.clear, bg="#333", fg="#ccc"
                  ).pack(fill="x", padx=10, pady=6)
        self.status = tk.Label(root, text="", font=("Consolas", 10), bg="#222", fg="#ddd",
                               justify="left", anchor="w", wraplength=300)
        self.status.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        root.bind("<space>", lambda e: self.stop())
        root.bind("<Escape>", lambda e: self.stop())
        self.tick()

    def stop(self):
        control.request(control.STOP)

    def park(self):
        control.request(control.STOP)
        control.request(control.PARK)

    def limp(self):
        control.request(control.STOP)
        control.request(control.LIMP)
        if not control.runner_alive():
            try:
                direct_limp()
                control.STATUS.write_text("LIMP sent directly to servos")
            except Exception as e:  # noqa: BLE001
                control.STATUS.write_text(f"direct LIMP failed: {e}")

    def clear(self):
        control.clear()

    def tick(self):
        alive = control.runner_alive()
        try:
            st = control.STATUS.read_text()
        except OSError:
            st = ""
        flags = " ".join(f.name.upper() for f in (control.STOP, control.PARK, control.LIMP) if f.exists())
        self.status.config(text=f"runner: {'ACTIVE' if alive else 'idle'}   {flags}\n{st}",
                           fg="#7f7" if alive and not flags else "#fc6" if flags else "#ddd")
        self.root.after(200, self.tick)


if __name__ == "__main__":
    root = tk.Tk()
    Panel(root)
    root.mainloop()
