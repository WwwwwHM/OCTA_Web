import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
      meta: { title: '首页' }
    },
    {
      path: '/history',
      name: 'history',
      component: () => import('../views/HistoryView.vue'),
      meta: { title: '历史记录' }
    },
    {
      path: '/file-manager',
      name: 'FileManager',
      component: () => import('../views/FileManager.vue'),
      meta: { title: '文件管理' }
    },
    {
      path: '/weight-manager',
      name: 'WeightManager',
      component: () => import('../views/WeightManager.vue'),
      meta: { title: '权重管理' }
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import('../views/AboutView.vue'),
      meta: { title: '关于' }
    },
  ],
})

export default router
