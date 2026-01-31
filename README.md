# Bootleg Text Slicer

Text transcription & slicing tool with visual timeline and WAV output. I made it to simplify slicing the CC0 audio files into words.

### Overview
The script makes it possible to:
- Transcribe an audio file into individual words.
- Display and interact with each word’s start and end positions on a visual timeline.
- Adjust timing offsets for the beginning and end of each word either globally or individually.
- Play full audio or specific words directly from within the app.
- Export words as separate `.wav` audio files.
- Record the timeline position, along with the global and per‑word timing offsets for each exported word, into a `cutTemplate.txt` file so that the individual words can later be played using only the source audio file.

SourceForge page: https://sourceforge.net/projects/bootleg-text-slicer/

![Alt Preview](https://github.com/Northstrix/bootleg-text-slicer/blob/main/preview.webp?raw=true)

Both scripts work, but I wouldn’t advise you to use `Bootleg Text Slicer V2.py` to transcribe more than 60–90 seconds at a time. Otherwise, its UI might become laggy. You can easily adjust the transcription duration by moving the start and end sliders below the timeline.

Successfully tested with English and Italian audio files.

AMD Athlon 3050U Performance:

    ============================================================
    BOOTLEG TEXT SLICER - TRANSCRIPTION COMPLETE
    File: piacevolinotti2_21_straparola_128kb.mp3
    Selected Range: 58.443s -> 117.814s (59.371s)
    Words Detected: 107
    Processing Time: 52.88s
    Efficiency: 1.12x Realtime
    ============================================================

File: [Notte Nona: FAVOLA I](https://www.archive.org/download/piacevolinotti2_1906_librivox/piacevolinotti2_21_straparola_128kb.mp3) from [Le Piacevoli Notti, Libro 2](https://librivox.org/le-piacevoli-notti-libro-2-by-giovanni-francesco-straparola/)

The `Bootleg Text Slicer V1.py` was made using [Google AI Studio](https://aistudio.google.com/) (Gemini 3 Flash Preview).

The `Bootleg Text Slicer V2.py` was created from the code of `Bootleg Text Slicer V1.py` using [Perplexity](https://www.perplexity.ai)
