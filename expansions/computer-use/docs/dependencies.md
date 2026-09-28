# Dependencies (modern baseline)

Last reviewed: 2026-09-06

## Official documentation checked

- Pydantic docs: https://docs.pydantic.dev/latest/
- Playwright Python docs: https://playwright.dev/python/docs/intro
- PyAutoGUI docs: https://pyautogui.readthedocs.io/en/latest/
- MSS docs: https://mss.readthedocs.io/en/stable/
- EasyOCR docs: https://www.jaided.ai/easyocr/

## Latest package versions snapshot (PyPI)

Collected via `https://pypi.org/pypi/<package>/json`:

- pydantic==2.13.5
- tenacity==9.1.4
- structlog==26.1.0
- httpx==0.28.1
- Pillow==12.3.0
- mss==10.2.0
- opencv-python==5.0.0.93
- easyocr==1.7.2
- playwright==1.62.0
- pyautogui==0.9.54
- pynput==1.8.2
- pywinauto==0.6.9

## Policy

- Use modern major lines (Pydantic v2+, Playwright latest stable).
- Keep desktop-specific dependencies optional when host/platform constraints apply.
- Re-check versions before each release tag.
