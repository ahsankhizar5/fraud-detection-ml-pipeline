// Main JavaScript for Fraud Detection System

// Global variables
let modelStatus = {
    trained: false,
    metrics: null
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Check model status on page load
    checkModelStatus();
    
    // Initialize tooltips
    initTooltips();
    
    // Add smooth scrolling
    addSmoothScrolling();
    
    // Add loading states to forms
    addLoadingStates();
});

// Check model status
function checkModelStatus() {
    fetch('/api/model-status')
        .then(response => response.json())
        .then(data => {
            modelStatus = data;
            updateModelStatusDisplay();
        })
        .catch(error => {
            console.error('Error checking model status:', error);
        });
}

// Update model status display
function updateModelStatusDisplay() {
    const statusElements = document.querySelectorAll('[data-model-status]');
    statusElements.forEach(element => {
        if (modelStatus.trained) {
            element.classList.remove('disabled');
            element.removeAttribute('disabled');
        } else {
            element.classList.add('disabled');
            element.setAttribute('disabled', 'disabled');
        }
    });
}

// Initialize Bootstrap tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Add smooth scrolling to anchor links
function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Add loading states to forms
function addLoadingStates() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.dataset.noLoading) {
                addLoadingState(submitBtn);
            }
        });
    });
}

// Add loading state to button
function addLoadingState(button) {
    const originalHTML = button.innerHTML;
    button.dataset.originalHtml = originalHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    button.disabled = true;
}

// Remove loading state from button
function removeLoadingState(button) {
    if (button.dataset.originalHtml) {
        button.innerHTML = button.dataset.originalHtml;
        button.disabled = false;
        delete button.dataset.originalHtml;
    }
}

// Show notification
function showNotification(message, type = 'info', duration = 5000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        max-width: 500px;
    `;
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Format percentage
function formatPercentage(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

// Animate counter
function animateCounter(element, start, end, duration = 1000) {
    const startTime = performance.now();
    const startValue = start;
    const endValue = end;
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease out)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        
        const currentValue = startValue + (endValue - startValue) * easeOut;
        element.textContent = Math.round(currentValue);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// Animate progress bar
function animateProgressBar(progressBar, targetWidth, duration = 1000) {
    const startTime = performance.now();
    const startWidth = 0;
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease out)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        
        const currentWidth = startWidth + (targetWidth - startWidth) * easeOut;
        progressBar.style.width = currentWidth + '%';
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// Copy text to clipboard
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Copied to clipboard!', 'success', 2000);
        }).catch(() => {
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
}

// Fallback copy function
function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showNotification('Copied to clipboard!', 'success', 2000);
    } catch (err) {
        showNotification('Failed to copy to clipboard', 'danger', 3000);
    }
    
    document.body.removeChild(textArea);
}

// Validate form fields
function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showFieldError(field, 'This field is required');
            isValid = false;
        } else {
            clearFieldError(field);
        }
    });
    
    return isValid;
}

// Show field error
function showFieldError(field, message) {
    clearFieldError(field);
    
    field.classList.add('is-invalid');
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    
    field.parentNode.appendChild(errorDiv);
}

// Clear field error
function clearFieldError(field) {
    field.classList.remove('is-invalid');
    const errorDiv = field.parentNode.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

// Debounce function
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        
        if (callNow) func.apply(context, args);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// Check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Add fade-in animation when elements come into view
function addScrollAnimations() {
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });
    
    animatedElements.forEach(element => {
        observer.observe(element);
    });
}

// Initialize scroll animations after DOM is loaded
document.addEventListener('DOMContentLoaded', addScrollAnimations);

// Handle network errors
function handleNetworkError(error) {
    console.error('Network error:', error);
    showNotification(
        'Network error occurred. Please check your connection and try again.',
        'danger',
        5000
    );
}

// API helper functions
const API = {
    // Generic API call function
    call: async function(endpoint, options = {}) {
        try {
            const response = await fetch(endpoint, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            handleNetworkError(error);
            throw error;
        }
    },
    
    // Get model status
    getModelStatus: function() {
        return this.call('/api/model-status');
    },
    
    // Train model
    trainModel: function() {
        return this.call('/api/train', { method: 'POST' });
    },
    
    // Predict transaction
    predictTransaction: function(data) {
        return this.call('/api/predict', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
};

// Export functions for use in other scripts
window.FraudDetection = {
    showNotification,
    formatNumber,
    formatCurrency,
    formatPercentage,
    animateCounter,
    animateProgressBar,
    copyToClipboard,
    validateForm,
    debounce,
    throttle,
    isInViewport,
    API,
    modelStatus
};