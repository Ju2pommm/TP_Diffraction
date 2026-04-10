#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Entry point for the Laser Diffraction Viewer application.

Run with:
    python main.py
"""

from app import DiffractionViewer


if __name__ == "__main__":
    app = DiffractionViewer()
    app.mainloop()
