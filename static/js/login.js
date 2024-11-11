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

// Toggle Sign Up & Sign In
// Sign Up & Sign In 버튼 클릭 시 표시할 섹션을 전환
document.getElementById('to_signup').addEventListener('click', () => {
    document.getElementById('login').style.display = 'none';
    document.getElementById('signup').style.display = 'block';
    window.location.hash = '#signup';
    window.scrollTo({ top: 0, behavior: 'smooth' }); // 스크롤을 맨 위로 이동
});

document.getElementById('to_login').addEventListener('click', () => {
    document.getElementById('signup').style.display = 'none';
    document.getElementById('login').style.display = 'block';
    window.location.hash = '#login';
    window.scrollTo({ top: 0, behavior: 'smooth' }); // 스크롤을 맨 위로 이동
});


function showSection(sectionId) {
    const loginSection = document.getElementById('login');
    const signupSection = document.getElementById('signup');
    if (sectionId === 'signup') {
        loginSection.style.display = 'none';
        signupSection.style.display = 'block';
    } else if (sectionId === 'login') {
        signupSection.style.display = 'none';
        loginSection.style.display = 'block';
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
