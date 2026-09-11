# Abdul Ghafoor Nizamani — Academic Website

Static academic website designed for GitHub Pages and postdoctoral applications. It uses plain HTML, CSS, and JavaScript with no build step and no third-party front-end dependencies.

## Local preview

From the repository root:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Core files

- `index.html` — page content and metadata
- `style.css` — responsive visual system
- `script.js` — mobile navigation, publication filters, and image lightbox
- `favicon.svg` — browser/site icon
- `photo.jpg` — profile photo
- `projects/` — publication thumbnails and full-resolution research figures

## Before publishing

1. Review all publication status wording and dates.
2. Add a current `CV.pdf` and a visible CV link once the final CV is ready.
3. Add the final GitHub Pages URL as a canonical URL and `og:url` after deployment.
4. Check the site on desktop and mobile.
5. Consider moving research code in `code/` to a dedicated repository or archival release if it is already available through GitHub/Zenodo; the academic website itself does not require those source files to render.

## GitHub Pages

In the GitHub repository, go to **Settings → Pages**, choose **Deploy from a branch**, select `main`, and use `/ (root)`.

No build command is required.
