# Academic Website (GitHub Pages)

Simple static site for a PhD candidate seeking postdoc positions. No build step needed — pure HTML/CSS.

## Local preview

Open `index.html` in a browser, or run:

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Publish on GitHub Pages

1. Create a new repository on GitHub (e.g. `username.github.io` for a site at `https://username.github.io`).
2. Upload/push these files (index.html, style.css, and optionally CV.pdf) to the repository.
3. Go to **Settings → Pages** and set **Source** to `Deploy from a branch` → `main` → `/ (root)`.
4. Your site is live at `https://username.github.io` within a minute.

## To-dos

- [ ] Replace `[YOUR NAME]` and other bracketed placeholders in `index.html`
- [ ] Fill in bio, research keywords, publications
- [ ] Add `CV.pdf`
- [ ] (Optional) Connect a custom domain in **Settings → Pages**
