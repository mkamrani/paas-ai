import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "Cool Demo PaaS",
  tagline: "Simple AWS Infrastructure as Code",
  favicon: "img/favicon.ico",

  future: {
    v4: true,
  },

  url: "https://your-docusaurus-site.example.com",
  baseUrl: "/",

  organizationName: "paas-ai",
  projectName: "cool-demo-paas",

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          routeBasePath: "/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: "img/docusaurus-social-card.jpg",
    navbar: {
      title: "Cool Demo PaaS",
      logo: {
        alt: "Cool Demo PaaS Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Documentation",
        },
        {
          href: "https://github.com/paas-ai/cool-demo-paas",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Documentation",
          items: [
            {
              label: "Getting Started",
              to: "/intro",
            },
            {
              label: "DSL Reference",
              to: "/dsl/ec2",
            },
          ],
        },
        {
          title: "AWS Services",
          items: [
            {
              label: "Networking",
              to: "/dsl/networking",
            },
            {
              label: "EC2",
              to: "/dsl/ec2",
            },
            {
              label: "ECS",
              to: "/dsl/ecs",
            },
            {
              label: "RDS",
              to: "/dsl/rds",
            },
            {
              label: "ALB",
              to: "/dsl/alb",
            },
            {
              label: "Route53",
              to: "/dsl/route53",
            },
            {
              label: "ACM",
              to: "/dsl/acm",
            },
          ],
        },
        {
          title: "Guidelines",
          items: [
            {
              label: "Best Practices",
              to: "/guidelines/best-practices",
            },
            {
              label: "Networking",
              to: "/guidelines/networking",
            },
            {
              label: "How-tos",
              to: "/guidelines/how-tos",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Cool Demo PaaS. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["yaml", "bash"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
