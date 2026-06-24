param(
    [Parameter(Mandatory = $true)]
    [string]$Title,

    [string[]]$Categories = @(),

    [string]$CategoriesCsv = "",

    [string[]]$Tags = @(),

    [string]$TagsCsv = "",

    [switch]$Math,

    [switch]$Mermaid,

    [string]$Image = "",

    [string]$Author = "wang",

    [string]$Timezone = "+0800",

    [string]$OutputDir = "_posts"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Convert-ToSlug {
    param([string]$Value)

    $slug = $Value.ToLowerInvariant()
    $slug = [regex]::Replace($slug, "\s+", "-")
    $slug = [regex]::Replace($slug, "[^a-z0-9\p{IsCJKUnifiedIdeographs}\-_]", "")
    $slug = [regex]::Replace($slug, "-{2,}", "-").Trim("-")

    if ([string]::IsNullOrWhiteSpace($slug)) {
        throw "Title could not be converted into a valid file slug."
    }

    return $slug
}

function Format-ArrayLiteral {
    param([string[]]$Values)

    $normalized = @(
        $Values |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )

    if ($normalized.Count -eq 0) {
        return "[]"
    }

    return "[" + ($normalized -join ", ") + "]"
}

function Split-CommaSeparatedValues {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return @()
    }

    return @(
        $Value.Split(",") |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}

if (-not [string]::IsNullOrWhiteSpace($CategoriesCsv)) {
    $Categories = Split-CommaSeparatedValues -Value $CategoriesCsv
}

if (-not [string]::IsNullOrWhiteSpace($TagsCsv)) {
    $Tags = Split-CommaSeparatedValues -Value $TagsCsv
}

if ($Categories.Count -eq 0) {
    throw "Please provide at least one topic category. Recommended categories: AI Agent, 大模型与训练, 检索与排序, 工具与框架, 编程基础, 博客建设."
}

$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path -Path (Get-Location) -ChildPath $OutputDir
}

if (-not (Test-Path -LiteralPath $resolvedOutputDir)) {
    New-Item -ItemType Directory -Path $resolvedOutputDir | Out-Null
}

$now = Get-Date
$datePart = $now.ToString("yyyy-MM-dd")
$dateTimePart = $now.ToString("yyyy-MM-dd HH:mm:ss")
$slug = Convert-ToSlug -Value $Title
$fileName = "$datePart-$slug.md"
$filePath = Join-Path -Path $resolvedOutputDir -ChildPath $fileName

if (Test-Path -LiteralPath $filePath) {
    throw "Post already exists: $filePath"
}

$frontMatter = @(
    "---"
    "layout: post"
    "title: `"$Title`""
    "author: `"$Author`""
    "date: $dateTimePart $Timezone"
    "categories: $(Format-ArrayLiteral -Values $Categories)"
    "tags: $(Format-ArrayLiteral -Values $Tags)"
)

if ($Math.IsPresent) {
    $frontMatter += "math: true"
}

if ($Mermaid.IsPresent) {
    $frontMatter += "mermaid: true"
}

if (-not [string]::IsNullOrWhiteSpace($Image)) {
    $frontMatter += "image: $Image"
}

$frontMatter += "---"

$body = @(
    ""
    "## One-Sentence Summary"
    ""
    "Summarize the main takeaway in 2-3 sentences. Answer what this post explains and why it matters."
    ""
    "## Background and Problem"
    ""
    "Describe the problem this topic solves and the scenarios where it usually appears."
    ""
    "## Core Concepts and Principles"
    ""
    "Break the notes into 2-4 focused subsections. Each subsection should explain one core idea."
    ""
    "### 1. Concept One"
    ""
    "Define the concept first, then explain why it matters."
    ""
    "### 2. Concept Two"
    ""
    "Explain how it relates to the first concept, how it differs, or when it applies."
    ""
    "## Key Details or Examples"
    ""
    "Place formulas, diagrams, code snippets, or examples from the original notes here and add the missing explanation."
    ""
    "## Practical Tips and Caveats"
    ""
    "- When this approach works well"
    "- Limits or pitfalls that are easy to miss"
    "- If there are alternatives, explain the tradeoff briefly"
    ""
    "## Summary"
    ""
    "Close with the main points and a practical conclusion."
    ""
)

$content = ($frontMatter + $body) -join [Environment]::NewLine
[System.IO.File]::WriteAllText($filePath, $content, [System.Text.UTF8Encoding]::new($false))

Write-Output $filePath
