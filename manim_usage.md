# Manim usage

1. Create a python file `.py` and save it, e.g. `HelloWorld.py`
2. Within the python file create a Scene (or InteractiveScene)`class`, e.g. `HelloWorld` which main method is `construct`

3. To run the code via shortcuts, use Terminus in the Sublime text editor:
	* "Ctrl + Shift + r": run the scene as a new scene consider the `n` lines up above your cursor location
	* "Ctrl + r": `manim_checkpoint_paste()` command allows to update the scene with the highlighted code. Highlight the piece of code
		you want to render and just press "Ctrl + r"

4. Otherwise, use the command: `manimgl HelloWorld.py HelloWorld -se 12`, where:
	* `manimgl` invoke the 3b1b class
	* `HelloWorld.py` is the python file to open
	* `HelloWorld` is the scene to render
	* `-se 12` is a flag that specify the number of rows (the first `12` rows) we want to render of the file

5. To export the scene creation use the command: `manimgl HelloWorld.py HelloWorld --prerun --finder -w -m`
	* `--prerun` allows to run the scene without seeing it
	* `--finder` specifies the default path in the finder
	* `-w` is the command to write the scene to a `.mp4` file
	* `-h/-m/-l` represent the render quality flag: high/medium/low resolution (default: `high`)

For more references see [3b1b Flags](https://3b1b.github.io/manim/getting_started/configuration.html)  