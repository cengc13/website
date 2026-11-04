// Initialize medium zoom.
// The theme background colour may be minified to a 3-digit hex (e.g. "#fff"),
// so appending an alpha suffix produces an invalid colour and a see-through
// overlay. Resolve it to rgb() first and build an explicit rgba() instead.
function zoomBackground() {
  var probe = document.createElement('div');
  probe.style.color = getComputedStyle(document.documentElement)
      .getPropertyValue('--global-bg-color').trim() || '#ffffff';
  document.body.appendChild(probe);
  var rgb = getComputedStyle(probe).color.match(/\d+/g) || [255, 255, 255];
  document.body.removeChild(probe);
  return 'rgba(' + rgb[0] + ', ' + rgb[1] + ', ' + rgb[2] + ', 0.97)';
}

$(document).ready(function() {
  medium_zoom = mediumZoom('[data-zoomable]', {
    background: zoomBackground(),
    margin: 24,
  });
  // Keep the overlay matching the theme when dark mode is toggled.
  medium_zoom.on('open', function() {
    medium_zoom.update({ background: zoomBackground() });
  });
});
