# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration of the OpenFAWE documentation (Read the Docs).

The pages of the user guide are cut from ``docs/user-guide.md`` - the single source, as published with the program - at
the start of every build (``build``): one page per chapter, the links between chapters made links between pages.
Nothing under ``guide/`` or ``_generated/`` is edited by hand."""
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
REPO = DOCS.parent

project = "OpenFAWE"
author = "Yinong Tian"
copyright = "2026, Yinong Tian and contributors"
version = release = "1.0.0"

extensions = ["myst_parser", "sphinx.ext.mathjax"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["colon_fence", "dollarmath", "html_image", "attrs_inline", "deflist"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "README.md"]
suppress_warnings = ["myst.header"]

html_theme = "sphinx_rtd_theme"
html_title = "OpenFAWE 1.0.0 documentation"
html_static_path = ["_static"]
html_css_files = ["openfawe.css"]
html_theme_options = dict(navigation_depth=3, collapse_navigation=False, sticky_navigation=True, titles_only=False,
                          style_external_links=True, prev_next_buttons_location="both")
html_context = dict(display_github=True, github_user="robytian", github_repo="OpenFAWE", github_version="main",
                    conf_py_path="/docs/site/")
html_show_sourcelink = False

GITHUB = "https://github.com/robytian/OpenFAWE/blob/main/"
PAGES = ("what-it-computes", "installation", "window", "examples", "replay", "export", "running", "output-files",
         "flight-modes", "validation", "limitations", "faq", "references")
GUIDES = dict(                                  # source, folder of the pages, the chapters' headings in page order
    en=("user-guide.md", "guide", ("## 1. What the program computes", "## 2. Installation", "## 3. The window",
                                   "## 4. Example flights", "## 5. Replay, wake and plots", "## 6. Export",
                                   "## 7. Running a simulation", "## 8. Output files", "## 9. The flight modes",
                                   "## Validation", "## 11. Scope and limitations", "## 12. Questions and answers",
                                   "## 13. References")),
    zh=("user-guide.zh.md", "zh", ("## 1. 程序算什么", "## 2. 安装", "## 3. 界面", "## 4. 算例", "## 5. 回放、尾迹与曲线",
                                   "## 6. 导出", "## 7. 自己计算", "## 8. 输出文件", "## 9. 飞行模式", "## 验证",
                                   "## 11. 适用范围与局限", "## 12. 常见问题", "## 13. 参考文献")))
RECORD_KEYS = dict(en=("record-keys.md", "reference/record-keys.md"), zh=("record-keys.zh.md", "zh/record-keys.md"))


def slug(heading):
    """The anchor of a Markdown heading as GitHub and MyST make it."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def rewrite(text, page_dir, where=None, page=None):
    """The links of a text of docs/ for its place in the site: chapters' anchors to their pages, the other documents
    to their pages, the media to the copy, anything else of the repository to GitHub."""
    up = "../" * page_dir.count("/") + "../"           # from the page's folder back to the site's root
    def anchor(m):
        target = None if where is None else where.get(m.group(1))
        if target is None or target == page:
            return f"](#{m.group(1)})"
        return f"]({target}.md#{m.group(1)})"
    text = re.sub(r"\]\(#([^)]+)\)", anchor, text)
    text = re.sub(r"\]\(user-guide\.md#([^)]+)\)", lambda m: f"]({up}guide/{GUIDE_WHERE['en'].get(m.group(1), 'what-it-computes')}.md#{m.group(1)})", text)
    text = re.sub(r"\]\(user-guide\.zh\.md#([^)]+)\)", lambda m: f"]({up}zh/{GUIDE_WHERE['zh'].get(m.group(1), 'what-it-computes')}.md#{m.group(1)})", text)
    text = text.replace("](user-guide.md)", f"]({up}guide/index.md)").replace("](user-guide.zh.md)", f"]({up}zh/index.md)")
    text = text.replace("](record-keys.md)", f"]({up}reference/record-keys.md)").replace("](record-keys.zh.md)", f"]({up}zh/record-keys.md)")
    text = text.replace("](../README.md#speed)", f"]({GITHUB}README.md#speed)").replace("](../README.md)", f"]({up}index.md)")
    text = re.sub(r"\]\(media/", f"]({up}_generated/media/", text)
    text = re.sub(r'src="media/', f'src="{up}_generated/media/', text)
    return text


GUIDE_WHERE = {}


def split(language):
    name, folder, headings = GUIDES[language]
    source = (DOCS/name).read_text(encoding="utf-8")
    starts = [(source.index("\n" + h + "\n") + 1, page) for h, page in zip(headings, PAGES)] + [(len(source), None)]
    pages = {page: source[a:b].rstrip().rstrip("-").rstrip() + "\n" for (a, page), (b, _) in zip(starts, starts[1:])}
    preamble = source[:starts[0][0]]
    where = {}
    for page, text in pages.items():
        for line in text.splitlines():
            m = re.match(r"^(#{2,4}) (.+)$", line)
            if m:
                where[slug(m.group(2))] = page
    GUIDE_WHERE[language] = where
    return folder, preamble, pages, where


def first_heading(text):
    return re.sub(r"^#+ ", "", text.splitlines()[0]).strip()


def build():
    for folder in ("guide", "zh", "reference", "_generated"):
        if (HERE/folder).exists():
            shutil.rmtree(HERE/folder)
    shutil.copytree(DOCS/"media", HERE/"_generated"/"media")
    parts = {language: split(language) for language in GUIDES}
    for language, (folder, preamble, pages, where) in parts.items():
        out = HERE/folder
        out.mkdir(parents=True, exist_ok=True)
        for page, text in pages.items():
            (out/f"{page}.md").write_text(rewrite(text, folder, where, page), encoding="utf-8")
        title = preamble.splitlines()[0]
        intro = "\n".join(line for line in preamble.splitlines()[1:] if line.strip() and not line.startswith(
            ("[English]", "**English**", "目录", "Contents")) and not re.match(r"^\d+\. \[", line) and line.strip() != "---")
        chapters = "\n".join(f"- [{first_heading(pages[page])}]({page}.md)" for page in PAGES)
        (out/"index.md").write_text(f"{title}\n\n{rewrite(intro, folder)}\n\n{chapters}\n", encoding="utf-8")
    for language, (name, target) in RECORD_KEYS.items():
        text = (DOCS/name).read_text(encoding="utf-8")
        place = HERE/target
        place.parent.mkdir(parents=True, exist_ok=True)
        place.write_text(rewrite(text, str(Path(target).parent)), encoding="utf-8")


build()


def _no_edit_link_on_cut_pages(app, pagename, templatename, context, doctree):
    """The pages cut from the guide are made at build time: their 'Edit on GitHub' would lead nowhere - the guide's own
    files (docs/user-guide.md, docs/user-guide.zh.md, docs/record-keys*.md) are the ones to edit."""
    if pagename.split("/")[0] in ("guide", "zh", "reference"):
        context["display_github"] = False


def setup(app):
    app.connect("html-page-context", _no_edit_link_on_cut_pages)
