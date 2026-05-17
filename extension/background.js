const ENDPOINT = "http://localhost:5000/predict";
const MENU_ID = "check-stego";

browser.runtime.onInstalled.addListener(() => {
  browser.contextMenus.create({
    id: MENU_ID,
    title: "Check for steganography",
    contexts: ["image"]
  });
});

browser.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== MENU_ID || !info.srcUrl) return;

  await notify("Analysing image…", "Sending to local detector", null);

  try {
    const imgResp = await fetch(info.srcUrl);
    if (!imgResp.ok) throw new Error(`Could not fetch image (HTTP ${imgResp.status})`);
    const blob = await imgResp.blob();

    const rawName = (info.srcUrl.split("/").pop() || "image.jpg").split("?")[0];
    const filename = rawName.includes(".") ? rawName : rawName + ".jpg";

    const form = new FormData();
    form.append("image", blob, filename);

    const predResp = await fetch(ENDPOINT, { method: "POST", body: form });
    const data = await predResp.json().catch(() => ({ error: "Server returned non-JSON" }));

    if (!predResp.ok || data.error) {
      throw new Error(data.error || `Server returned HTTP ${predResp.status}`);
    }

    const title = data.is_stego
      ? "⚠ Steganography detected"
      : "✓ No steganography detected";
    const message = `${data.model} · ${data.confidence}% confidence`;
    await notify(title, message, data.is_stego);
  } catch (err) {
    const hint = /NetworkError|Failed to fetch/i.test(err.message)
      ? "Is the Flask server running on localhost:5000?"
      : err.message;
    await notify("Detection failed", hint, null);
  }
});

function notify(title, message, isStego) {
  return browser.notifications.create({
    type: "basic",
    iconUrl: "data:image/svg+xml;utf8," + encodeURIComponent(iconSvg(isStego)),
    title,
    message
  });
}

function iconSvg(isStego) {
  const color = isStego === true ? "#dc2626" : isStego === false ? "#16a34a" : "#4a7bff";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
    <circle cx="48" cy="48" r="44" fill="${color}"/>
    <text x="48" y="62" text-anchor="middle" font-size="48" font-family="sans-serif" fill="white">🛡</text>
  </svg>`;
}
