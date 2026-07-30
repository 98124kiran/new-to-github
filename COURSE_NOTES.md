# GitHub & Markdown Course Notes

## Course Overview
This course provides a comprehensive introduction to GitHub and version control fundamentals, communication through Markdown, and deploying content with GitHub Pages.

---

## Module 1: Getting Started - 3 Core Steps

### Introduction
Before diving into GitHub, it's important to understand the three fundamental steps that form the foundation of version control:
1. Initialize your repository
2. Make changes and commit them
3. Collaborate through pull requests

### Step 1: Understanding Repositories
A **repository (repo)** is the central hub of your project:
- Contains all your files and folders
- Keeps track of every version and change (version history)
- Acts as a complete history of your project's evolution
- Can be public (open to everyone) or private (restricted access)

**Key Concepts:**
- **Files**: The actual content of your project
- **Folders**: Organizational structure for grouping files
- **Version Control**: Automatic tracking of who changed what and when

### Step 2: Working with Branches
A **branch** is a parallel version of your repository:
- The **main** branch is the definitive version of your project
- Think of branching as creating a copy of your repository to work on
- You can work independently on a branch without affecting the main branch
- Once your work is ready, you merge it back to main

**Branch Workflow:**
1. Create a new branch from main
2. Make your changes on the new branch
3. Test and review your changes
4. Merge the branch back to main when ready

**Best Practices:**
- Use descriptive branch names that reflect what you're working on
- Keep branches focused on a single feature or fix
- Delete branches after merging to keep the repository clean

### Step 3: Understanding Commits
A **commit** is a snapshot of changes:
- Records a set of changes made to files or folders
- Each commit includes:
  - The changes made
  - A commit message describing what changed
  - Author information
  - Timestamp
  - A unique identifier (SHA hash)

**Commit Guidelines:**
- Write clear, descriptive commit messages
- Keep commits focused on a single change or feature
- Commit frequently to maintain a detailed history
- Make commit messages in present tense (e.g., "Add user authentication")

---

## Module 2: Collaboration with Pull Requests

### What is a Pull Request?
A **pull request (PR)** is a proposal to merge changes from one branch to another:
- Allows team members to review your changes before merging
- Enables discussion about the changes
- Ensures code quality through peer review
- Creates a record of why changes were made

**Pull Request Workflow:**
1. Create a branch and make your changes
2. Push the branch to GitHub
3. Open a pull request comparing your branch to main
4. Add a descriptive title and description
5. Team members review and comment
6. Address feedback with additional commits
7. Merge the pull request once approved

**PR Best Practices:**
- Keep PRs small and focused on a single feature
- Write a clear description of what the PR changes
- Reference related issues
- Respond to reviewer feedback promptly
- Test your changes before requesting review

---

## Module 3: Communication Using Markdown

### What is Markdown?
Markdown is a lightweight markup language for creating formatted text:
- Easy to read and write
- Converts to HTML and other formats
- Perfect for documentation, README files, and communication
- Supported by GitHub for all text content

### Markdown Syntax

#### Headings
```markdown
# Heading 1 (H1)
## Heading 2 (H2)
### Heading 3 (H3)
#### Heading 4 (H4)
##### Heading 5 (H5)
###### Heading 6 (H6)
```

#### Text Formatting
```markdown
*italic* or _italic_
**bold** or __bold__
***bold italic***
~~strikethrough~~
```

#### Lists

**Unordered Lists:**
```markdown
- Item 1
- Item 2
  - Nested item 2.1
  - Nested item 2.2
- Item 3
```

**Ordered Lists:**
```markdown
1. First item
2. Second item
   1. Nested item 2.1
   2. Nested item 2.2
3. Third item
```

**Checklists:**
```markdown
- [x] Completed task
- [ ] Incomplete task
- [x] Another completed task
```

#### Links and Images
```markdown
[Link text](https://example.com)
[Link with title](https://example.com "Title")

![Alt text](image-url.jpg)
![Alt text](image-url.jpg "Image title")
```

#### Code
**Inline Code:**
```markdown
Use `variable_name` for inline code
```

**Code Blocks:**
````markdown
```python
def hello_world():
    print("Hello, World!")
```
````

#### Blockquotes
```markdown
> This is a blockquote
> 
> > This is a nested blockquote
```

#### Horizontal Rule
```markdown
---
***
___
```

#### Tables
```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
```

### Markdown Use Cases
- **README.md**: Project documentation and setup instructions
- **CONTRIBUTING.md**: Guidelines for contributing to the project
- **Pull Request Descriptions**: Explain your changes
- **Issue Descriptions**: Clearly describe bugs or feature requests
- **GitHub Pages**: Create project websites
- **Documentation**: Create comprehensive documentation sites

---

## Module 4: GitHub Pages

### What is GitHub Pages?
GitHub Pages is a service that allows you to:
- Host static websites directly from your GitHub repository
- Use it for project documentation, portfolios, blogs, and more
- Automatically deploy changes when you push to your repository
- Use custom domains or GitHub-provided subdomains

### Types of GitHub Pages Sites

**User/Organization Sites:**
- One site per user or organization
- Repository must be named `username.github.io`
- Published from main branch

**Project Sites:**
- One site per repository (except for User/Organization repositories)
- Repository can have any name
- Published from a designated branch or directory

### Setting Up GitHub Pages

1. **Create a repository** (or use existing one)
2. **Create an index.html file** (or use Jekyll for markdown-based sites)
3. **Enable GitHub Pages** in repository settings
4. **Choose the publishing source** (main branch, gh-pages branch, or /docs folder)
5. **Access your site** at `https://username.github.io` or `https://username.github.io/repo-name`

### Using Jekyll with GitHub Pages

GitHub Pages supports Jekyll, which allows you to:
- Write content in Markdown
- Use templates and themes
- Build static sites easily

**Basic Jekyll Setup:**
1. Create a `_config.yml` file
2. Use markdown files for content
3. Choose a theme
4. Push to GitHub

### Example GitHub Pages Workflow

```markdown
# Project Name

Welcome to my GitHub Pages site!

## Features
- Feature 1
- Feature 2
- Feature 3

## Getting Started
1. Clone the repository
2. Follow the setup instructions
3. Deploy to GitHub Pages
```

---

## Module 5: GitHub Workflow Best Practices

### Development Workflow
1. **Create an issue** describing the feature or bug
2. **Create a branch** from main with a descriptive name
3. **Make changes** and commit frequently with clear messages
4. **Push your branch** to GitHub
5. **Create a pull request** with a detailed description
6. **Request review** from team members
7. **Address feedback** with additional commits
8. **Merge to main** once approved
9. **Delete the branch** to keep the repository clean

### Naming Conventions

**Branch Names:**
- Use lowercase and hyphens: `feature/user-authentication`
- Use prefixes: `feature/`, `bugfix/`, `hotfix/`, `docs/`
- Include issue number if applicable: `feature/123-user-auth`

**Commit Messages:**
- Use imperative mood: "Add" not "Added" or "Adds"
- Capitalize the first letter
- Keep the subject line under 50 characters
- Add a detailed body if needed, separated by a blank line

**Examples:**
- ✅ "Add user authentication system"
- ✅ "Fix memory leak in database connection"
- ✅ "Update documentation for API endpoints"
- ❌ "fixed stuff" (vague and not capitalized)
- ❌ "This commit adds a new feature for user auth" (too long)

### Code Review Best Practices
- Be respectful and constructive in feedback
- Ask questions rather than making demands
- Suggest specific improvements
- Acknowledge good work
- Keep discussions focused on the code

---

## Quick Reference

### Essential Git Commands
```bash
# Create a new branch
git checkout -b branch-name

# Stage changes
git add file-name
git add .

# Commit changes
git commit -m "Your commit message"

# Push to remote
git push origin branch-name

# Pull latest changes
git pull origin main

# View status
git status

# View commit history
git log
```

### GitHub Terminology
| Term | Definition |
|------|-----------|
| Repository | Project folder containing files and version history |
| Branch | Parallel version of your repository |
| Commit | Snapshot of changes with a message |
| Pull Request | Proposal to merge changes into main |
| Fork | Personal copy of someone else's repository |
| Clone | Download a repository to your local machine |
| Push | Upload local commits to GitHub |
| Pull | Download updates from GitHub |
| Merge | Combine changes from one branch to another |

---

## Summary

### Key Takeaways
1. **Repositories** store your project and track its history
2. **Branches** allow parallel development without affecting main
3. **Commits** create meaningful snapshots of your work
4. **Pull Requests** enable collaborative review and merging
5. **Markdown** is the standard for documentation in GitHub
6. **GitHub Pages** lets you host static websites for free
7. **Clear communication** through good commit messages and PR descriptions is essential

### Next Steps
- Practice creating branches and commits
- Write your first pull request
- Create a README.md with Markdown
- Set up a GitHub Pages site for your project
- Contribute to open-source projects

---

## Resources

### Official Documentation
- [GitHub Docs](https://docs.github.com)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Pages Documentation](https://pages.github.com)
- [Markdown Guide](https://www.markdownguide.org)

### Learning Platforms
- GitHub Learning Lab
- Codecademy Git Course
- Udemy Git and GitHub courses
- Pluralsight version control paths

---

**Last Updated:** 2026-05-21

This course provides the foundation for effective version control, collaboration, and project documentation using GitHub and Markdown. Happy coding!
