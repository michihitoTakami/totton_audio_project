"""USB input -> I2S output bridge for Raspberry Pi (Pi5).

Issue #823: PC(USB Audio) -> Pi5 -> I2S(TX, master) -> Jetson.

このパッケージは既存の RTP 系とは独立しており、ALSA のデバイス切断や
サンプルレート切替(44.1/48系)を検知して自動復帰するためのランナーを提供する。
"""
