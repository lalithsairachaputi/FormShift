document.addEventListener('DOMContentLoaded', () => {
  const passwordInput = document.getElementById('password');
  const passwordToggle = document.getElementById('passwordToggle');
  const eyeIcon = passwordToggle.querySelector('.eye-icon');
  const form = document.getElementById('loginForm');
  const passkeyButton = document.getElementById('passkeyButton');
  const submitButton = form.querySelector('.submit-button');
  const workspaceUrl = '../main_page/index.html';

  const eyeOpenPath = `
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
    <circle cx="12" cy="12" r="3"></circle>
  `;

  const eyeClosedPath = `
    <path d="M17.94 17.94A10.94 10.94 0 0 1 12 20c-7 0-11-8-11-8a21.6 21.6 0 0 1 5.06-6.06"></path>
    <path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 8 11 8a21.6 21.6 0 0 1-2.16 3.19"></path>
    <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24"></path>
    <line x1="1" y1="1" x2="23" y2="23"></line>
  `;

  // Toggle password visibility
  passwordToggle.addEventListener('click', () => {
    const isPassword = passwordInput.type === 'password';
    passwordInput.type = isPassword ? 'text' : 'password';
    eyeIcon.innerHTML = isPassword ? eyeClosedPath : eyeOpenPath;
    passwordToggle.classList.toggle('active', isPassword);
    passwordToggle.setAttribute('aria-label', isPassword ? 'Hide password' : 'Show password');
  });

  // Quick passkey sign-in path (fast lane, no typing required)
  passkeyButton.addEventListener('click', () => {
    passkeyButton.disabled = true;
    passkeyButton.querySelector('span').textContent = 'Waiting for device...';

    setTimeout(() => {
      passkeyButton.querySelector('span').textContent = 'Signed in';
      window.location.href = workspaceUrl;
    }, 700);
  });

  // Email/password submit
  form.addEventListener('submit', (e) => {
    e.preventDefault();

    const email = document.getElementById('email').value.trim();
    const password = passwordInput.value;

    if (!email || !password) {
      shakeForm();
      return;
    }

    submitButton.textContent = 'Logging in...';
    submitButton.disabled = true;

    // Short delay to feel snappy rather than heavy
    setTimeout(() => {
      submitButton.textContent = 'Logged in';
      window.location.href = workspaceUrl;
    }, 500);
  });

  function shakeForm() {
    form.style.transition = 'transform 0.08s ease';
    let count = 0;
    const shake = setInterval(() => {
      form.style.transform = count % 2 === 0 ? 'translateX(-4px)' : 'translateX(4px)';
      count++;
      if (count > 5) {
        clearInterval(shake);
        form.style.transform = 'translateX(0)';
      }
    }, 40);
  }
});