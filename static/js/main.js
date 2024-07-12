function togglePassword(inputId, icon) {
    var input = document.getElementById(inputId);
    if (input.type === "password") {
        input.type = "text";
        icon.classList.remove('fa-lock');
        icon.classList.add('fa-unlock');
    } else {
        input.type = "password";
        icon.classList.remove('fa-unlock');
        icon.classList.add('fa-lock');
    }
}


document.addEventListener('DOMContentLoaded', () => {
    const registerForm = document.getElementById('register-form');

    registerForm.addEventListener('submit', async function(e) {
        e.preventDefault(); // Prevent the form from submitting immediately

        const username = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('pass').value;
        const confirmPassword = document.getElementById('cpass').value;

        if (!validateEmail(email)) {
            alert('Email address format is invalid');
            return;
        }

        if (password !== confirmPassword) {
            alert('Password and Confirm Password are not the same');
            return;
        }

        // Proceed to check the email with the server
        checkEmailWithServer({username:username ,  email:email , password:password});
    });
});

function validateEmail(email) {
    const re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(String(email).toLowerCase());
}




async function checkEmailWithServer(data) {
    try {
        const response = await fetch('/check-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: data.email, username: data.username, password: data.password }),
        });

        const res = await response.json();
        if (!res.valid) {
            alert(res.message);
        }else if (res.valid && res.messageSent) {
            alert('Welcome email successfully sent!');
            window.location.href = '/login';
        } else {
            alert('Email is valid, but the welcome message could not be sent.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while verifying the email: ' + error.message);
    }
}












//Forget Email on submit
document.addEventListener('DOMContentLoaded', () => {
    const forgetForm = document.getElementById('forget-form');

    forgetForm.addEventListener('submit', async function(e) {
        e.preventDefault(); // Prevent the form from submitting immediately

        const email = document.getElementById('email').value;
        
        if (!validateEmail(email)) {
            alert('Email address format is invalid');
            return;
        }

        
        // Proceed to check the email with the server
        checkEmail({ email:email });
    });
});


async function checkEmail(data) {
    try {
        const response = await fetch('/check-forget-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: data.email}),
        });

        const res = await response.json();
        if (!res.valid) {
            alert('Email doesnot exists!');
        }else if (res.valid && res.messageSent) {
            alert('OTP will Expire after 30 second!');
            // Redirect to '/reset' route with email as query parameter
            window.location.href = `/reset?email=${encodeURIComponent(data.email)}`;
        } else {
            alert('Email is valid, but the OTP message could not be sent.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while verifying the email: ' + error.message);
    }
}




//Resend
document.addEventListener('DOMContentLoaded', () => {
    const resendBtn = document.getElementById('resend-btn');
    const email = document.getElementById('email').dataset.email;

    resendBtn.addEventListener('click', async function() {
        checkEmail({ email: email });
    });
});
