self.addEventListener("install", e => {
    console.log("Service Worker Installed");
});

self.addEventListener("push", event => {
    const data = event.data.json();

    self.registration.showNotification("💊 Medicine Reminder", {
        body: data.message,
        icon: "icon.png"
    });
});