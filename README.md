# U.S. Agricultural Commodity Price Dashboard

A research portfolio site displaying a Power BI report about corn, soybean and wheat prices across business cycles, 1995–2025.

## Setup
1. Publish your Power BI report to Power BI Service.
2. Obtain an appropriate embed URL. **Publish to web makes report data public**; only use for data you are authorized to disclose. For private reports use secure embedding and configure viewer permissions/licenses.
3. In `index.html`, replace the empty `POWER_BI_EMBED_URL` value with the URL in the Power BI iframe's `src` attribute.
4. Replace `REPLACE_USERNAME/REPLACE_REPOSITORY` in the GitHub button link with your repository address.
5. Push `index.html` to your GitHub repository root.
6. In GitHub: Settings > Pages > Deploy from a branch > main > /(root) > Save.

## Suggested repository layout
- `index.html`: public website
- `scripts/`: Python analysis code
- `dashboard/`: Power BI project (only if licensed and safe to share)
- `data/`: public datasets or documented data sources
- `figures/`: research figures

## Important
This template does not contain the user's report or real GitHub link until configured.
