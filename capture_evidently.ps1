# --------------------------------------------------------------
# capture_evidently.ps1 – saves Evidently report as a PNG
# --------------------------------------------------------------
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class WinAPI {
    [DllImport("user32.dll")]
    public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    public struct RECT {
        public int Left; public int Top; public int Right; public int Bottom;
    }
}
"@
Add-Type -AssemblyName System.Drawing

# ---- Settings -------------------------------------------------
$repoRoot   = "D:\ray_train_cca1"
$screensDir = Join-Path $repoRoot "screenshots"
$outFile    = Join-Path $screensDir "evidently_report.png"

# ---- Launch Evidently HTML in Chrome --------------------------
Start-Process "chrome.exe" -ArgumentList "file:///$($repoRoot.Replace('\','/') + '/evidently_report.html')"
Start-Sleep -Seconds 3   # give Chrome a moment to start

# ---- Find the Chrome window that contains the file name -----
$expected = "*evidently_report.html* - Google Chrome"
$chrome = Get-Process -Name chrome -ErrorAction SilentlyContinue |
          Where-Object { $_.MainWindowTitle -like $expected } |
          Select-Object -First 1

if (-not $chrome) {
    Write-Error "Could not locate the Chrome tab showing Evidently. Make sure the HTML opened and is the active tab."
    exit 1
}

# ---- Bring that window to the front -------------------------
$hwnd = $chrome.MainWindowHandle
[WinAPI]::SetForegroundWindow($hwnd) | Out-Null
Start-Sleep -Milliseconds 500   # let the OS repaint the window

# ---- Get its screen rectangle -------------------------------
$rect = New-Object WinAPI+RECT
[WinAPI]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
$width  = $rect.Right  - $rect.Left
$height = $rect.Bottom - $rect.Top

# ---- Capture the bitmap -------------------------------------
$bmp = New-Object System.Drawing.Bitmap $width, $height
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bmp.Size)

# ---- Save the screenshot ------------------------------------
$bmp.Save($outFile, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Host "`n✅ Evidently screenshot saved to:" -ForegroundColor Green
Write-Host $outFile
