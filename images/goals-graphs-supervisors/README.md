# Image sources

## Hero image

`rivera-detroit-industry-hero.jpg` is cropped from Carptrash's photograph of
Diego Rivera's *Detroit Industry* (1932–33),
[Rivera frescos 02 on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Rivera_frescos_02.jpg).
The [original photograph](https://upload.wikimedia.org/wikipedia/commons/e/e2/Rivera_frescos_02.jpg)
is licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
This cropped version carries the same license. Credit: Diego Rivera (mural),
Carptrash (photograph); cropped and resized for this post.
The [Detroit Institute of Arts](https://dia.org/collection/58539) identifies
the mural as public domain when published in the United States.
Source and status checked on 13 November 2026.

The original photograph is 5,590 by 3,729 pixels. The crop uses the rectangle
`(0, 800, 5590, 2270)` in original-image coordinates, with an exclusive right
and bottom edge. It is resized to 2,000 by 526 pixels and saved as an optimized
progressive JPEG. No recoloring was applied. The crop retains the foreground
workers' faces and hands while removing the lower foliage.
The post caption credits the artist and photographer, links the source and
license and identifies the crop.

## Previous hero candidate

`steel-valley-hero.jpg` is cropped from Louis Lozowick's 1936 lithograph
[Steel Valley on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Steel_Valley_by_Louis_Lozowick,_lithograph.jpg).
The file page marks it public domain in the United States under
`PD-US-no notice`; this does not assert public-domain status in every country.
Source and status checked on 13 November 2026.

The original is 4,239 by 2,980 pixels. The crop uses the rectangle
`(100, 650, 4140, 1713)` in original-image coordinates, resized to 2,000 by
526 pixels and saved as a progressive JPEG. No recoloring was applied.
The post caption credits the artist, links the source and identifies the crop.

## Original diagram

`supervisor-first.svg` illustrates the division of attention described in the
post: blue parent nodes show Michael's supervisory focus, while orange child
nodes show the agents' task focus. It is an original diagram for this post,
not a screenshot of a running system.

`goal-feedback-loop.svg` is an original diagram of the review-process experiment.
It distinguishes automated comparisons from the user's assessment and shows how
that feedback guides the next attempt. Neither diagram reports a measured gain.

`dynamic-workflow.svg` is an original explanatory diagram based on
[Anthropic's dynamic workflow documentation](https://code.claude.com/docs/en/workflows.md),
checked on 13 November 2026. It is not an official Anthropic illustration.
The main agent writes the workflow; a separate runtime executes its JavaScript.
Intermediate results stay in script variables, and the final output returns to
the main conversation. A script may call additional agents for synthesis.

## Verus demonstration

`verus-demo.png` is an unmodified copy of the demonstration shown in the
[Verus README](https://github.com/verus-lang/verus#readme), downloaded on
13 November 2026 from
[`source/docs/verus-demo.png`](https://raw.githubusercontent.com/verus-lang/verus/main/source/docs/verus-demo.png).
It shows verification errors in a terminal and VS Code for a minimum function
that returns `y` in both branches and fails its specified postcondition.
The repository's MIT notice is retained in
[`verus-demo.LICENSE.txt`](verus-demo.LICENSE.txt).

## Author-supplied cartoon

`dilbert-dashboard.gif` is the unmodified *Dilbert* dashboard strip supplied by
Michael for inclusion in the post. The image retains its original Scott Adams
and syndication attribution, dated 2007. It was downloaded from the
[author-supplied LinkedIn image URL](https://media.licdn.com/dms/image/v2/D4E22AQHvymXxHcojwQ/feedshare-shrink_1280/feedshare-shrink_1280/0/1713878058455?e=1796256000&v=beta&t=GVGCywAp4MRZHlGvBOt9N8gTdSQpvPbrIq1RLpThWrk)
on 13 November 2026. That URL is time-limited; the post uses the local file.
This records provenance and the author's inclusion request, not a verified
redistribution license.

## Screenshot sources

These screenshots are still frames from the projects' public demonstration
recordings, not captures of Michael's or Delphos Labs' working environments.
No private repositories, dashboard configuration or company work appears in them.

| File | Source | Frame (zero-based) |
| --- | --- | --- |
| `gh-dash.png` | [gh-dash overview](https://github.com/dlvhdr/gh-dash/blob/e6752f8f2cf10f650d4f6e136385ba70f57d11f1/docs/src/assets/overview.gif) | 0 |
| `tuicr.png` | [tuicr demonstration](https://github.com/agavra/tuicr/blob/f098fb8f0911f6b9b14c80e23f79ab084cf5ef91/public/tuicr-demo.gif) | 127 |

The frames were converted to RGB PNG without cropping or changing their content.

## gh-dash license

Source: [LICENSE.txt](https://github.com/dlvhdr/gh-dash/blob/e6752f8f2cf10f650d4f6e136385ba70f57d11f1/LICENSE.txt).

MIT License

Copyright (c) 2025 Dolev Hadar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## tuicr license

Source: [LICENSE](https://github.com/agavra/tuicr/blob/f098fb8f0911f6b9b14c80e23f79ab084cf5ef91/LICENSE).

MIT License

Copyright (c) 2025 tuicr contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
