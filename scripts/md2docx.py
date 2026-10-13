"""md (with ```latex blocks) -> docx with native equations, via the pandoc binary from pypandoc_binary."""
import re, subprocess, sys, tempfile, pypandoc
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
s = re.sub(r"```latex\n(.*?)\n```", lambda m: "$$\n" + m.group(1) + "\n$$", s, flags=re.S)
with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
    f.write(s)
subprocess.run([pypandoc.get_pandoc_path(), f.name, "-o", out], check=True)
