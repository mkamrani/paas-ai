import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'DSL Reference',
      items: [
        'dsl/networking',
        'dsl/ec2',
        'dsl/ecs',
        'dsl/rds',
        'dsl/alb',
        'dsl/route53',
        'dsl/acm',
      ],
    },
    {
      type: 'category',
      label: 'Guidelines',
      items: [
        'guidelines/best-practices',
        'guidelines/networking',
        'guidelines/how-tos',
      ],
    },
  ],
};

export default sidebars;
