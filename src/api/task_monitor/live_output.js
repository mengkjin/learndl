// Streamlit calls the renderer again when data changes. Keep the viewport in place.
export default function ({ data, parentElement }) {
    const viewport = parentElement.querySelector('.live-output');
    const lines = viewport.firstElementChild;
    const jump = parentElement.querySelector('button');
    const state = viewport.liveOutputState ??= { following: true, top: 0 };

    function setTop(top) {
        viewport.scrollTop = top;
        state.top = viewport.scrollTop;
    }

    viewport.onscroll = () => {
        // A content update can also emit a scroll event.
        if (viewport.scrollTop === state.top) return;
        state.top = viewport.scrollTop;
        state.following = viewport.scrollHeight - viewport.clientHeight - state.top <= 2;
        jump.hidden = state.following;
    };
    jump.onclick = () => {
        state.following = true;
        setTop(viewport.scrollHeight);
        jump.hidden = true;
        viewport.focus({ preventScroll: true });
    };

    if (state.content === data.content && state.style === data.style) return;

    const previous = Array.from(lines.children);
    const anchor = state.following ? -1 : previous.findIndex(
        line => line.offsetTop + line.offsetHeight > viewport.scrollTop,
    );
    const offset = anchor < 0 ? 0 : viewport.scrollTop - previous[anchor].offsetTop;
    const incoming = document.createElement('div');
    if (data.style === 'colored') {
        // Only parse_live_output's allowlisted, escaped markup reaches this component.
        incoming.innerHTML = data.content;
    } else {
        for (const text of data.content.split('\n')) {
            const line = document.createElement('div');
            line.textContent = text;
            incoming.append(line);
        }
    }

    // Match the retained suffix of the old tail against the new prefix.
    let retained = -1;
    if (anchor >= 0 && state.style === data.style) {
        const oldLines = previous.map(line => line.innerHTML);
        const newLines = Array.from(incoming.children, line => line.innerHTML);
        for (let start = 0; start <= anchor; start++) {
            const overlap = Math.min(oldLines.length - start, newLines.length);
            // The final old line can be incomplete and change on the next write.
            const stable = Math.min(overlap, Math.max(1, oldLines.length - start - 1));
            if (anchor - start < overlap && oldLines.slice(start, start + stable)
                .every((line, index) => line === newLines[index])) {
                retained = anchor - start;
                break;
            }
        }
    }
    lines.replaceChildren(...incoming.children);
    state.content = data.content;
    state.style = data.style;
    if (state.following) {
        setTop(viewport.scrollHeight);
    } else {
        setTop(retained < 0 ? 0 : lines.children[retained].offsetTop + offset);
    }
    jump.hidden = state.following;
}
