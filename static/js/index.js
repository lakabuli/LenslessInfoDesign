document.addEventListener('DOMContentLoaded', function () {
  var burger = document.querySelector('.navbar-burger');

  if (!burger) {
    return;
  }

  var target = document.getElementById(burger.getAttribute('data-target'));

  burger.addEventListener('click', function () {
    var isActive = burger.classList.toggle('is-active');

    if (target) {
      target.classList.toggle('is-active', isActive);
    }

    burger.setAttribute('aria-expanded', String(isActive));
  });
});
