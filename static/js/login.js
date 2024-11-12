// Label Positioning Functions
function labelUp(input) {
    input.parentElement.children[0].setAttribute("class", "change_label");
}

function labelDown(input) {
    if (input.value.length === 0) {
        input.parentElement.children[0].classList.remove("change_label");
    }
}

// Show & Hide Password
document.getElementById('eye_icon_signup').addEventListener('click', () => {
    togglePasswordVisibility('eye_icon_signup', 'signup_password');
});

document.getElementById('eye_icon_login').addEventListener('click', () => {
    togglePasswordVisibility('eye_icon_login', 'login_password');
});

function togglePasswordVisibility(iconId, inputId) {
    const eye_icon = document.getElementById(iconId);
    const password = document.getElementById(inputId);
    if (eye_icon.classList.contains("fa-eye-slash")) {
        eye_icon.classList.remove('fa-eye-slash');
        eye_icon.classList.add('fa-eye');
        password.setAttribute('type', 'text');
    } else {
        eye_icon.classList.remove('fa-eye');
        eye_icon.classList.add('fa-eye-slash');
        password.setAttribute('type', 'password');
    }
}

// Handle hash change on page load and URL change
window.addEventListener('load', handleHashChange);
window.addEventListener('hashchange', handleHashChange);

function handleHashChange() {
    const targetId = window.location.hash.substring(1);
    if (targetId === 'signup' || targetId === 'login') {
        showSection(targetId);
    }
}
