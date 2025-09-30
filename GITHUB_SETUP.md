# GitHub Repository Setup Guide

This guide helps you upload your DDoS detection project to GitHub.

## Before You Start

Make sure you have:

- [ ] A GitHub account
- [ ] Git installed on your computer
- [ ] All project files ready in your local directory

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the repository details:
   - **Repository name**: `ddos-detection-cnn-sampling` (or your preferred name)
   - **Description**: "DDoS attack detection using counter-based sampling and CNN"
   - **Visibility**: Choose Public or Private
   - **Initialize**: Do NOT initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 2: Prepare Your Local Repository

Open a terminal/command prompt in your project directory and run:

```bash
# Initialize git repository
git init

# Add all files to staging
git add .

# Make your first commit
git commit -m "Initial commit: Counter-based sampling and CNN DDoS detection project"

# Add the remote repository (replace YOUR_USERNAME and YOUR_REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

## Step 3: Verify Upload

1. Go to your GitHub repository page
2. Check that all files are uploaded:
   - [ ] README.md (main project description)
   - [ ] requirements.txt (Python dependencies)
   - [ ] .gitignore (files to ignore)
   - [ ] LICENSE (license information)
   - [ ] CONTRIBUTING.md (contribution guidelines)
   - [ ] Counter-Based Sampling/ folder with code
   - [ ] CNN_DDoS_detection/ folder with code

## Step 4: Repository Settings (Optional)

### Add Topics/Tags

1. Go to your repository page
2. Click the gear icon next to "About"
3. Add relevant topics: `ddos-detection`, `cnn`, `deep-learning`, `network-security`, `pcap`, `sampling`

### Enable Issues and Discussions

1. Go to Settings tab
2. Scroll to "Features" section
3. Enable "Issues" for bug reports and feature requests
4. Enable "Discussions" for community Q&A

### Create Releases

1. Go to "Releases" on the main repository page
2. Click "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: `Initial Release`
5. Describe the release features

## Step 5: Repository Description

Make sure your repository has a good description. Edit the "About" section to include:

- Short description: "DDoS attack detection using counter-based sampling and CNN"
- Website: (if you have one)
- Topics: ddos-detection, cnn, deep-learning, network-security, pcap-analysis

## Recommended Repository Structure

Your final GitHub repository should look like this:

```
your-repo-name/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License information
├── CONTRIBUTING.md             # Contribution guidelines
├── GITHUB_SETUP.md            # This file
├── Counter-Based Sampling/     # Part 1: Sampling algorithm
│   ├── README.md              # Sampling documentation
│   └── Counter-Based Sampling.py
└── CNN_DDoS_detection/        # Part 2: CNN detection
    ├── README.md              # Original LUCID docs
    ├── lucid_cnn.py          # Main CNN implementation
    ├── lucid_dataset_parser.py
    ├── util_functions.py
    └── sample-dataset/        # (excluded by .gitignore)
```

## Common Issues and Solutions

### Authentication Issues

If you get authentication errors:

- Use GitHub CLI: `gh auth login`
- Or use personal access token instead of password
- Or use SSH keys for authentication

### Large Files

If you have large PCAP files:

- They should be excluded by .gitignore
- Consider using Git LFS for large files if needed
- Upload sample data separately or provide download links

### Permission Denied

Make sure you have write access to the repository and correct remote URL.

## Next Steps After Upload

1. **Add a good repository description** on GitHub
2. **Star your own repository** to make it easier to find
3. **Share the repository link** with collaborators
4. **Consider adding badges** to README.md for build status, license, etc.
5. **Set up GitHub Actions** for automated testing (optional)

## Example Git Commands for Updates

After the initial upload, use these commands for updates:

```bash
# Check status
git status

# Add specific files
git add filename.py

# Add all changes
git add .

# Commit changes
git commit -m "Descriptive commit message"

# Push to GitHub
git push

# Pull latest changes
git pull
```

## Repository URL

Once created, your repository will be available at:
`https://github.com/YOUR_USERNAME/YOUR_REPO_NAME`

## Troubleshooting

If you encounter issues:

1. Check GitHub's documentation
2. Verify your internet connection
3. Ensure correct repository permissions
4. Try using GitHub Desktop as an alternative to command line

Good luck with your GitHub upload!
