#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;

uniform samplerExternalOES u_texture;

uniform vec2 u_texelSize;
uniform vec4 u_keyColor;

uniform mat3 uNdcToSurface;   // NDC [-1,1]^2 -> surface (s,t) in [0,1]^2
uniform vec2 uViewportSize;   // framebuffer size in pixels
uniform int  uHasSurface;     // 1 if homography is valid, 0 otherwise

in vec2 v_TexCoord;
out vec4 fragColor;

void main() {
    // Original camera sample (used for final output if not keyed)
    vec4 texColor = texture(u_texture, v_TexCoord);

    vec2 ndc = (gl_FragCoord.xy / uViewportSize) * 2.0 - 1.0;

    bool insideSurface = false;
    if (uHasSurface == 1) {
        vec3 q = uNdcToSurface * vec3(ndc, 1.0);
        float w = q.z;
        if (abs(w) > 1e-6) {
            vec2 st = q.xy / w;  // surface coords

            // Small padding in surface coordinates (e.g., 1–2% beyond each edge)
            float margin = 0.02; // tweak: 0.01–0.03 depending on how much leeway you want

            insideSurface =
            st.x >= -margin && st.x <= 1.0 + margin &&
            st.y >= -margin && st.y <= 1.0 + margin;
        }
    }

    // If we are outside the surface, bypass chroma key completely.
    if (!insideSurface) {
        fragColor = texColor;
        return;
    }

    // -------- PRE-KEY BLUR (5-tap cross) --------
    // Blur only for classification to stabilize hue/sat under YUV subsampling noise.
    vec3 rgbC = texColor.rgb;
    vec3 rgbR = texture(u_texture, v_TexCoord + vec2( u_texelSize.x, 0.0)).rgb;
    vec3 rgbL = texture(u_texture, v_TexCoord + vec2(-u_texelSize.x, 0.0)).rgb;
    vec3 rgbU = texture(u_texture, v_TexCoord + vec2(0.0,  u_texelSize.y)).rgb;
    vec3 rgbD = texture(u_texture, v_TexCoord + vec2(0.0, -u_texelSize.y)).rgb;

    vec3 rgb = (rgbC * 0.40) + (rgbR + rgbL + rgbU + rgbD) * 0.15;
    // -------------------------------------------

    float luminance = dot(rgb, vec3(0.299,0.587,0.114));

    // RGB -> HSV (compact)
    float Cmax = max(rgb.r, max(rgb.g, rgb.b));
    float Cmin = min(rgb.r, min(rgb.g, rgb.b));
    float delta = Cmax - Cmin;

    float H = 0.0;
    if (delta > 1e-5) {
        if (Cmax == rgb.r) H = mod((rgb.g - rgb.b)/delta, 6.0);
        else if (Cmax == rgb.g) H = ((rgb.b - rgb.r)/delta) + 2.0;
        else H = ((rgb.r - rgb.g)/delta) + 4.0;
        H *= 60.0; // degrees
        if (H < 0.0) H += 360.0;
    }

    float S = (Cmax <= 1e-5) ? 0.0 : delta / Cmax;
    float V = Cmax;

    // Hue windows (degrees). Tune if needed.
    bool hueGreen = (H > 70.0 && H < 170.0);
    bool hueBlue  = (H > 190.0 && H < 260.0);
    bool hueBridge = (H >= 165.0 && H <= 205.0);

    // Bright vs shadow handled by V and S instead of absolute channels
    bool isGreen = hueGreen && S > 0.20 && V > 0.12;
    bool isBlue  = hueBlue  && S > 0.15 && V > 0.10;
    bool isBridge = hueBridge && S > 0.10 && V > 0.10;

    // Low-sat shadow fallback: if very dark, allow weaker saturation
    if (luminance < 0.35) {
        isGreen = hueGreen && S > 0.6 && V > 0.08;
        isBlue  = hueBlue  && S > 0.5 && V > 0.07;

        isBridge = hueBridge && S > 0.25 && V > 0.07;
    }

    const vec4 KEY_COLOR = vec4(0.0, 1.0, 0.0, 1.0); // pure green, opaque

    if (isGreen || isBlue || isBridge) fragColor = u_keyColor;
    else fragColor = texColor; // IMPORTANT: keep original unblurred camera color
}
