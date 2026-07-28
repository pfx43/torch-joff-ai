# 本地编译 docs/*.tex 为 PDF（XeLaTeX + ctex，中文）
# 用法：
#   pwsh docs\编译PDF.ps1                      # 编译 论文方法完整设计.tex
#   pwsh docs\编译PDF.ps1 -TexFile 其他文件.tex  # 编译指定文件
#   pwsh docs\编译PDF.ps1 -Clean               # 编译后清理中间文件
#
# 依赖（本机已装好，2026-07-26 验证）：
#   MiKTeX  %LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64   (已在用户 PATH)
#   Perl    D:\Strawberry\perl\bin                          (latexmk 需要)

param(
    [string]$TexFile = "论文方法完整设计.tex",
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$docs = $PSScriptRoot
$miktex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
$perl = "D:\Strawberry\perl\bin"

foreach ($p in @($miktex, $perl)) {
    if (-not (Test-Path $p)) { throw "缺少依赖目录：$p" }
    if ($env:PATH -notlike "*$p*") { $env:PATH = "$p;$env:PATH" }
}

$src = Join-Path $docs $TexFile
if (-not (Test-Path $src)) { throw "找不到源文件：$src" }

# 中间文件放到临时目录，保持仓库干净；只把 PDF 取回 docs/
$build = Join-Path $env:TEMP "paper_tex_build"
New-Item -ItemType Directory -Force -Path $build | Out-Null

Push-Location $docs
try {
    # latexmk 自动跑足够趟数，解决目录与交叉引用
    & latexmk -xelatex -interaction=nonstopmode -file-line-error -outdir="$build" $TexFile
    $ok = ($LASTEXITCODE -eq 0)
} finally {
    Pop-Location
}

$base = [IO.Path]::GetFileNameWithoutExtension($TexFile)
$log = Join-Path $build "$base.log"
$pdf = Join-Path $build "$base.pdf"

if (Test-Path $log) {
    $t = Get-Content $log -Raw
    function Count-Pat($pat) { ([regex]::Matches($t, $pat)).Count }
    Write-Output ""
    Write-Output "---- 编译质量检查 ----"
    Write-Output ("错误              : " + (Count-Pat '(?m)^(?:\S+\.tex:\d+:|! )'))
    Write-Output ("中文缺字          : " + (Count-Pat 'Missing character'))
    Write-Output ("未定义交叉引用    : " + (Count-Pat 'Reference `[^'']*'' on page .* undefined'))
    Write-Output ("PDF 书签非法记号  : " + (Count-Pat 'Token not allowed'))
    Write-Output ("超出页宽 Overfull : " + (Count-Pat 'Overfull'))
    if ($t -match '\((\d+) pages') { Write-Output ("页数              : " + $Matches[1]) }
}

if ($ok -and (Test-Path $pdf)) {
    Copy-Item $pdf (Join-Path $docs "$base.pdf") -Force
    Write-Output ""
    Write-Output "成功：$base.pdf 已生成在 docs\"
} else {
    Write-Output ""
    Write-Output "编译失败。完整日志：$log"
    exit 1
}

if ($Clean) {
    Remove-Item $build -Recurse -Force
    Write-Output "已清理中间文件。"
}
