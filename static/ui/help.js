/* Help buttons used by the classic workbench. Load as a classic script. */
(function (global) {
  'use strict';

  let instance = null;

  function init() {
    if (instance) return instance;
    if (!document.body) {
      document.addEventListener('DOMContentLoaded', init, { once: true });
      return null;
    }

    let popover = null;
    let activeTip = null;
    let pointerFocusTip = null;
    let observer = null;
    const listeners = [];

    function on(target, name, handler, options) {
      target.addEventListener(name, handler, options);
      listeners.push([target, name, handler, options]);
    }

    function tipFrom(target) {
      const element = target?.nodeType === 1 ? target : target?.parentElement;
      return element?.closest?.('.wb-help-tip') || null;
    }

    function prepare(tip) {
      if (!tip || !tip.matches('.wb-help-tip')) return;
      if (tip.tagName === 'BUTTON') tip.type = 'button';
      if (!tip.hasAttribute('aria-label') && tip.dataset.note) {
        tip.setAttribute('aria-label', '说明：' + tip.dataset.note);
      }
      tip.setAttribute('aria-expanded', tip === activeTip ? 'true' : 'false');
    }

    function ensurePopover() {
      if (!popover) {
        popover = document.createElement('div');
        popover.className = 'wb-help-popover';
        popover.id = 'h3-help-popover';
        popover.setAttribute('role', 'tooltip');
        popover.hidden = true;
        document.body.append(popover);
      }
      return popover;
    }

    function hide() {
      if (activeTip) {
        activeTip.setAttribute('aria-expanded', 'false');
        activeTip.removeAttribute('aria-describedby');
      }
      activeTip = null;
      if (popover) popover.hidden = true;
    }

    function show(tip) {
      if (!tip?.isConnected || !tip.matches('.wb-help-tip')) return;
      const note = tip.dataset.note || '';
      if (!note) return;
      if (activeTip && activeTip !== tip) hide();
      const box = ensurePopover();
      prepare(tip);
      activeTip = tip;
      tip.setAttribute('aria-expanded', 'true');
      tip.setAttribute('aria-describedby', box.id);
      box.textContent = note;
      box.hidden = false;
      const rect = tip.getBoundingClientRect();
      const width = box.offsetWidth;
      const height = box.offsetHeight;
      box.style.left = Math.max(12, Math.min(global.innerWidth - width - 12, rect.left + rect.width / 2 - width / 2)) + 'px';
      box.style.top = Math.max(8, rect.top - height - 8 >= 8 ? rect.top - height - 8 : Math.min(global.innerHeight - height - 8, rect.bottom + 8)) + 'px';
    }

    function compact(root) {
      if (!root || root.nodeType !== 1) return;
      const hints = [
        ...(root.matches('.hint,.script-help,.wb-help') ? [root] : []),
        ...root.querySelectorAll('.hint,.script-help,.wb-help'),
      ];
      for (const hint of hints) {
        if (!hint.isConnected || hint.closest('button,summary,label,a') ||
            hint.querySelector('a,button,input,select,textarea,img,video') ||
            !hint.textContent.trim()) continue;
        const tip = document.createElement('button');
        tip.type = 'button';
        tip.className = 'wb-help-tip ui-help-tip';
        tip.textContent = '?';
        tip.dataset.note = hint.textContent.trim();
        tip.setAttribute('aria-label', '说明：' + tip.dataset.note);
        prepare(tip);
        const field = hint.parentElement?.classList.contains('field') ? hint.parentElement : null;
        const sibling = hint.previousElementSibling;
        const anchor = field?.querySelector('label') || (sibling?.matches('h2,h3,h4,summary,label') ? sibling : null);
        if (anchor) {
          anchor.append(tip);
          hint.remove();
        } else {
          hint.replaceWith(tip);
        }
      }
      if (root.matches('.wb-help-tip')) prepare(root);
      root.querySelectorAll('.wb-help-tip').forEach(prepare);
    }

    on(document, 'pointerdown', event => { pointerFocusTip = tipFrom(event.target); }, true);
    on(document, 'pointerover', event => {
      if (event.pointerType === 'touch') return;
      const tip = tipFrom(event.target);
      if (tip) show(tip);
    });
    on(document, 'pointerout', event => {
      if (event.pointerType === 'touch') return;
      const tip = tipFrom(event.target);
      if (tip && tip === activeTip && !tip.contains(event.relatedTarget) &&
          !popover?.contains(event.relatedTarget) && document.activeElement !== tip) hide();
    });
    on(document, 'pointerleave', event => {
      if (event.target === popover && document.activeElement !== activeTip) hide();
    }, true);
    on(document, 'focusin', event => {
      const tip = tipFrom(event.target);
      if (tip && pointerFocusTip !== tip) show(tip);
      pointerFocusTip = null;
    });
    on(document, 'focusout', event => {
      if (tipFrom(event.target) === activeTip) hide();
    });
    on(document, 'click', event => {
      const tip = tipFrom(event.target);
      if (!tip) { hide(); return; }
      event.preventDefault();
      event.stopPropagation();
      if (activeTip === tip) hide();
      else show(tip);
      pointerFocusTip = null;
    }, true);
    on(document, 'keydown', event => { if (event.key === 'Escape') hide(); });
    on(document, 'scroll', hide, true);

    compact(document.body);
    observer = new MutationObserver(changes => {
      for (const change of changes) {
        for (const node of change.addedNodes) if (node.nodeType === 1) compact(node);
      }
      if (activeTip && !activeTip.isConnected) hide();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    instance = {
      compact,
      show,
      hide,
      destroy() {
        observer.disconnect();
        listeners.forEach(([target, name, handler, options]) => target.removeEventListener(name, handler, options));
        hide();
        popover?.remove();
        instance = null;
      },
    };
    return instance;
  }

  global.H3Help = { init };
})(window);
