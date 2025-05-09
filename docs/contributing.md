## Contributing Guidelines for FedEasy

### Introduction

FedEasy is an open-source federated learning framework that aims to make advanced machine learning techniques more accessible to a wider audience. We welcome contributions from the community to help improve and expand the framework. This document outlines the guidelines for contributing to FedEasy.

### Getting Started

1. **Familiarize yourself with the codebase**: Before contributing, take some time to review the FedEasy codebase and understand the overall architecture and design.
2. **Choose an issue to work on**: Browse the [issue tracker](https://github.com/nclabteam/FedEasy/issues) to find an issue that interests you and aligns with your skills, or create a new issue to work on.
3. **Fork the repository**: On the FedEasy GitHub page, click the "Fork" button at the top right corner to create your own copy of the repository.

### Cloning and Setting Up Your Local Environment

1. **Clone the repository**: Clone your forked repository to your local machine using the following command, replacing `<your-username>` with your GitHub username:
   ```bash
   git clone https://github.com/<your-username>/FedEasy.git
   ```
2. **Navigate to the cloned repository**:
   ```bash
   cd FedEasy
   ```

### Creating a New Branch

1. **Fetch the latest changes from the upstream repository**:
   ```bash
   git fetch upstream
   ```
2. **Create a new branch from the `main` branch**:
   ```bash
   git checkout -b feature/add-my-feature upstream/main
   ```

### Making Your Changes

1. **Implement your changes**: Ensure you follow the code style guidelines, add tests, and update documentation as needed.
2. **Stage your changes**:
   ```bash
   git add .
   ```
3. **Commit your changes**: Commit your changes with a descriptive commit message following the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format:
   ```bash
   git commit -m "feat: add my awesome feature"
   ```

### Pushing Your Branch

1. **Push your branch to your forked repository**:
   ```bash
   git push origin feature/add-my-feature
   ```

### Creating a Pull Request

1. **Visit the FedEasy repository on GitHub**: Go to the FedEasy repository on GitHub.
2. **Create a new pull request**: Click the "New pull request" button. Select your forked repository and branch.
3. **Provide a clear description**: In the pull request description, clearly explain the problem you are addressing, the changes you've made, and how those changes address the problem. Include any relevant screenshots or code snippets.

### Review Process

1. **Code review**: The FedEasy maintainers will review your contribution to ensure it meets the contribution guidelines and is consistent with the codebase.
2. **Feedback and revisions**: You may receive feedback and requests for revisions during the review process. You can make additional commits to your branch and push them to your forked repository; they will automatically be added to the pull request.
3. **Merge**: Once your pull request is approved, it will be merged into the `main` branch. Congratulations, you've successfully contributed to FedEasy!

### Additional Resources

* **FedEasy documentation**: [FedEasy Documentation on Read the Docs](https://fedeasy.readthedocs.io/)
* **FedEasy issue tracker**: [FedEasy issue tracker](https://github.com/nclabteam/FedEasy/issues)
* **FedEasy community forum**: [FedEasy community forum](https://github.com/nclabteam/FedEasy/discussions)

### Acknowledgments

We appreciate your contributions to FedEasy and look forward to working with you to make the framework better.

### License

By contributing to FedEasy, you agree to license your contributions under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

By following these guidelines, you can contribute effectively to the FedEasy project and help make it a more robust and feature-rich federated learning framework.