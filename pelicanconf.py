#!/usr/bin/env python
# -*- coding: utf-8 -*- #

# Site settings
AUTHOR = 'Shubham Jain'
SITENAME = 'My Personal Blog'
SITEURL = ''
SITESUBTITLE = 'Thoughts, tutorials, and industry insights'
SITEDESCRIPTION = 'A personal blog about programming, data science, and technology'

# Content path
PATH = 'content'

# Locale settings
TIMEZONE = 'America/Los_Angeles'
DEFAULT_LANG = 'en'
DEFAULT_DATE_FORMAT = '%B %d, %Y'

# Feed generation (disabled for development)
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Theme configuration
THEME = 'themes/elegant'

# Social media links
SOCIAL = (
    ('GitHub', 'https://github.com/jainshubham23'),
    ('Twitter', 'https://twitter.com/jainshubham23'),
    ('LinkedIn', 'https://www.linkedin.com/in/shubham-jain2310/'),
)

# Menu configuration
DISPLAY_PAGES_ON_MENU = True
DISPLAY_CATEGORIES_ON_MENU = True

# URL structure
ARTICLE_URL = 'posts/{date:%Y}/{date:%m}/{slug}.html'
ARTICLE_SAVE_AS = 'posts/{date:%Y}/{date:%m}/{slug}.html'
PAGE_URL = '{slug}.html'
PAGE_SAVE_AS = '{slug}.html'
CATEGORY_URL = 'category/{slug}.html'
CATEGORY_SAVE_AS = 'category/{slug}.html'
TAG_URL = 'tag/{slug}.html'
TAG_SAVE_AS = 'tag/{slug}.html'

# Archive settings
ARCHIVES_SAVE_AS = 'archives.html'
TAGS_SAVE_AS = 'tags.html'
CATEGORIES_SAVE_AS = 'categories.html'
YEAR_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/index.html'
MONTH_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/{date:%m}/index.html'

# Static files
STATIC_PATHS = ['images', 'static']

# Elegant theme specific settings
LANDING_PAGE_TITLE = 'Welcome to My Blog'
PROJECTS_TITLE = 'My Projects'
PROJECTS_INTRO = 'Here are some projects I have been working on'

# SEO and metadata
ROBOTS = 'index, follow'
SUMMARY_MAX_LENGTH = 150

# Pagination
DEFAULT_PAGINATION = 10

# Code highlighting
PYGMENTS_STYLE = 'github'

# Copyright
COPYRIGHT_YEAR = 2025
COPYRIGHT_NAME = 'Shubham Jain'

# Optional: Uncomment for development with relative URLs
# RELATIVE_URLS = True

# Optional: Analytics and comments (configure as needed)
# GOOGLE_ANALYTICS = 'UA-XXXXXXX-X'
# DISQUS_SITENAME = 'your-disqus-sitename'