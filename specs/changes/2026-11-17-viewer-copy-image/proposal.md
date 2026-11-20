# Per-panel image copy

Add a copy-image button to each method panel. Capture the current viewport, selected particles and cycle interval as a crisp PNG with a white background, method title, axes and ticks, normalized rho and cycle interval. Image text stays French, matching existing figures.

Copy the PNG through the browser clipboard API on the user's click. Keep unavailable or loading results disabled, report success only after a successful write, and offer the same image as a download when clipboard access is unavailable or denied.

Update the existing `gc2d-main-reflected-rho-div20` website in place. Preserve the numerical dataset, backgrounds and historical websites.
