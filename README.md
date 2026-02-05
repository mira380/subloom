## Subloom

Subloom is a fully local subtitle generator for audio and video.

1. Extracts audio from video
2. Preprocesses audio
3. Runs ASR (Kotoba or other engines)
4. Double checks with a second ASR model
5. Post-processes text (kanji normalization and cleanup)
6. Outputs ".srt" and transcript files


## Designed For

- Anime
- YouTube videos
- Language learning & mining


## Requirements

### Required
- Linux (Windows should work but not tested)
- Python 3.10+
- FFmpeg
- whisper.cpp
- An ASR engine
  - e.g. **Kotoba Whisper**

### Optional (Recommended)
- A GPU (huge speed improvement)
- **Ollama** (used for kanji/text cleanup)


## How to Use

Installation guide in progress (maybe)

### ▶ Process a Single File

python subloom.py run "Episode 01.mkv" --ollama

**You can also add certain styles that you want and even batch process a folder . i.e.**

python subloom.py run "Episode 01.mkv" --ollama --ollama-style anime

python subloom.py batch "/path/to/anime/folder" --ollama

### URL

python subloom.py run "https://youtu.be/VIDEO_ID" --ollama

## Future Features

- Optimize code and time needed to generate subtitles ✅ (can be improved upon further)

- Improve audio capturing 🟡 (improve the engine or rewrite until its what i want)

- Put in protections that prevents LLM from changing meaning/nuance and keeps original meaning clear ✅ (should be working)

- Simplify usage of commands/program itself ❌ (not high priority currently)

- Improve subtitle timing ⏳

- Flag lines the model is unsure about ⏳

- Smarter LLM checks/intelligence upgrades ⏳

- Less clutter when outputting completed files 🟡 (need to organize a little better)

### Target Goal

- **15 minutes** for ~2 hours of content on a mid/high-end PC using ollama.

- Acheving 98%+ subtitle accuracy (meaning, reading, nuance all in tact)

---

## Notes

Literally a personal project and not refined at all. I just wanted something that fit my workflow and couldn’t find anything that worked, so here we are.

Highkey vibe coded af 😭🥀
